"""
The data contract — the single internal shape every component reads and writes.

Everything hangs off this: the checks write SignalOutputs into a CaseRecord,
the combiner reads them, the router sets `decision`, the swipe app updates
`human_verdict`, and the CaseRecord IS the SQLite row. When real Shopee data
arrives we write ONE loader that maps their format into CaseRecord; nothing
else changes.

SCORING DIRECTION IS LOCKED: every score is CREDIBILITY (0.0 = looks fraudulent,
1.0 = looks trustworthy). Never risk. A missing/inapplicable signal contributes
NEUTRAL, never suspicion.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import json

# A signal that cannot run contributes this to the combined score — neutral,
# never suspicious. This is the single most important invariant in the system.
NEUTRAL_SCORE = 0.5

# The five checks. The first four vote on fraud signals; the Defender argues the
# claim is legitimate and pulls credibility UP (it points opposite the others).
SIGNAL_NAMES = ("authenticity", "completeness", "tamper", "relevance", "defender")

# Decision routing labels.
DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"
DECISION_ESCALATE = "escalate"
DECISION_RESUBMIT = "resubmit"

# RESUBMIT is auto-decided, NOT a human-review route. It exists because Rung 0's
# quality verdict is already final: a black or unfocused frame is a complete
# determination, and a reviewer looking at the same black rectangle learns nothing
# the pixel math didn't already know. Queueing it burns a review slot for zero
# information and — worse — creates a payout path, because a reviewer with no
# signal and a queue to clear tends to wave the case through. So "unreadable"
# leaves the human queue entirely.
#
# It is deliberately not REJECT. Both routes cost the same (zero reviewers) and
# both deny the fraudster a payout, so automation and fraud-leak closure do not
# distinguish them; the only difference is what happens to an honest buyer with a
# cheap camera in bad light. Reject denies that claim on evidence that was never
# judged. Resubmit asks for a readable file and keeps the claim alive. Same
# savings, without spending the false-positive budget on the poorest users.
#
# Decisions that count as automated (i.e. consume no human review).
AUTO_DECISIONS = (DECISION_APPROVE, DECISION_REJECT, DECISION_RESUBMIT)

# Reason codes carried alongside a decision. Rung 0 (deterministic pre-validation)
# writes these so the swipe app / analytics can tell *why* a case was routed the
# way it was without re-deriving it. Only DUPLICATE_PROOF is a fraud signal that
# auto-rejects; unreadable media routes to RESUBMIT (quality failure != fraud —
# an honest buyer with a bad phone camera must never be auto-rejected).
REASON_DUPLICATE_PROOF = "duplicate_proof"        # reused/near-identical media -> reject
REASON_CORRUPTED_FILE = "corrupted_file"          # won't decode -> resubmit (ops)
REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # too blurry/dark/small/short -> resubmit
REASON_PASSED_PREVALIDATION = "passed_prevalidation"    # cleared Rung 0, went to the agents

# Rung 1a — Authenticity runs solo, first. If it alone meets the strong-fraud veto
# (see scoring.VETO_SCORE/VETO_CONFIDENCE), the case rejects immediately and the
# remaining four checks (Rung 1b) are never called — saves 4 VLM calls on the
# clearest fraud. Distinct from Rung 0's duplicate_proof so logs/swipe-app can
# tell "the single hardest check was dispositive" apart from "hash-matched reuse".
REASON_AUTHENTICITY_DISPOSITIVE = "authenticity_dispositive_fraud"


@dataclass
class Evidence:
    """One piece of submitted proof (references a media file by name/path)."""
    filename: str                 # e.g. "237872216204114.mp4"
    kind: str = "unknown"         # "image" | "video" | "unknown"
    path: Optional[str] = None    # local path once the media is located/downloaded

    # Filled by Rung 0 (pre-validation), all deterministic / zero API cost.
    phash: Optional[str] = None   # perceptual hash (hex), for reuse/duplicate detection
    exif_editor: Optional[str] = None  # editing-software fingerprint from EXIF, if any.
                                       # A soft PRIOR passed into the Authenticity agent —
                                       # NOT a gate. Absence is neutral (screenshots strip EXIF).

    @staticmethod
    def infer_kind(filename: str) -> str:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext in ("mp4", "mov", "avi", "mkv", "webm"):
            return "video"
        if ext in ("jpg", "jpeg", "png", "gif", "bmp", "webp"):
            return "image"
        return "unknown"


@dataclass
class ItemListing:
    """One purchased item in the order (an order can contain several)."""
    shop_id: str
    item_id: str
    listing_link: Optional[str] = None
    price_php: Optional[float] = None   # listing price (range midpoint), PHP
    title: Optional[str] = None


@dataclass
class SignalOutput:
    """
    One check's vote. `score` is CREDIBILITY in [0,1] (high = trustworthy).
    If the check can't run, set applicable=False; the combiner then treats it
    as NEUTRAL and it never lowers credibility.
    """
    signal_name: str
    score: float = NEUTRAL_SCORE      # credibility 0..1
    verdict: str = "unknown"          # short human tag, e.g. "clean" / "pre-opened"
    reason_string: str = ""           # plain-language justification for the brief
    confidence: float = 0.0           # how sure THIS check is of its own score, 0..1
    applicable: bool = True           # False => contributes NEUTRAL, never suspicion

    def effective_score(self) -> float:
        """Score the combiner should use: real score if applicable, else neutral."""
        return self.score if self.applicable else NEUTRAL_SCORE


@dataclass
class CaseRecord:
    """
    One return case = one Order ID (which may span several items and several
    pieces of evidence). This dataclass IS the SQLite row.
    """
    case_id: str
    session_id: str
    return_reason: str = ""
    items: list[ItemListing] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    video_frames: list[str] = field(default_factory=list)   # extracted frame paths
    submitted_at: Optional[str] = None

    # Claim value = order value proxy (sum of item listing prices), PHP. None when
    # no item carries a price — the economic layer routes such cases to a human.
    # In production, replace with the actual requested refund amount.
    claim_value_php: Optional[float] = None

    # Per-agent output slots, keyed by signal_name.
    signals: dict[str, SignalOutput] = field(default_factory=dict)

    # Filled by the combiner / router / synthesis.
    credibility_score: Optional[float] = None   # 0..100 (higher = more trustworthy)
    decision: Optional[str] = None              # approve | reject | escalate
    reason_code: Optional[str] = None           # why (esp. Rung-0 routes); see REASON_*
    brief: str = ""                             # synthesis agent's plain-language note
    runtime_ms: Optional[int] = None

    # Rung-0 (pre-validation) trace: per-evidence QC metrics + gate outcome. Kept
    # for the brief / swipe-app explanation and for tuning thresholds vs the dataset.
    prevalidation: Optional[dict] = None

    # Expected-loss (pricing) layer trace: the EscalationDecision audit dict —
    # claim value, net exposure, both expected losses, break-even threshold, route,
    # and reason. Surfaced in the reviewer UI so a human sees WHY it was routed.
    economic: Optional[dict] = None

    # Filled later.
    human_verdict: Optional[str] = None         # swipe app writes this back
    true_label: Optional[str] = None            # ground truth, if available
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- convenience ---------------------------------------------------------

    @property
    def has_video(self) -> bool:
        return any(e.kind == "video" for e in self.evidence)

    @property
    def has_image(self) -> bool:
        return any(e.kind == "image" for e in self.evidence)

    def set_signal(self, sig: SignalOutput) -> None:
        self.signals[sig.signal_name] = sig

    # --- serialization: the nested detail lives in signals_json --------------

    def signals_json(self) -> str:
        """The full per-signal breakdown + brief + evidence/items, as a JSON blob.

        Stored in the single TEXT column `signals_json`; we only ever read it
        back whole, never query into it.
        """
        payload = {
            "items": [asdict(i) for i in self.items],
            "evidence": [asdict(e) for e in self.evidence],
            "video_frames": self.video_frames,
            "submitted_at": self.submitted_at,
            "signals": {k: asdict(v) for k, v in self.signals.items()},
            "brief": self.brief,
            "reason_code": self.reason_code,
            "prevalidation": self.prevalidation,
            "claim_value_php": self.claim_value_php,
            "economic": self.economic,
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_row(cls, row: dict) -> "CaseRecord":
        """Rebuild a CaseRecord from a SQLite row dict."""
        blob = json.loads(row.get("signals_json") or "{}")
        rec = cls(
            case_id=row["case_id"],
            session_id=row["session_id"],
            return_reason=row.get("return_reason") or "",
            items=[ItemListing(**i) for i in blob.get("items", [])],
            evidence=[Evidence(**e) for e in blob.get("evidence", [])],
            video_frames=blob.get("video_frames", []),
            submitted_at=blob.get("submitted_at"),
            signals={k: SignalOutput(**v) for k, v in blob.get("signals", {}).items()},
            credibility_score=row.get("credibility_score"),
            decision=row.get("decision"),
            reason_code=blob.get("reason_code"),
            brief=blob.get("brief", ""),
            prevalidation=blob.get("prevalidation"),
            claim_value_php=blob.get("claim_value_php"),
            economic=blob.get("economic"),
            runtime_ms=row.get("runtime_ms"),
            human_verdict=row.get("human_verdict"),
            true_label=row.get("true_label"),
            created_at=row.get("created_at") or datetime.now(timezone.utc).isoformat(),
        )
        return rec
