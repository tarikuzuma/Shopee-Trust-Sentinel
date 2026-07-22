"""
Scoring combiner + decision router.

Turns the five per-check SignalOutputs into a single credibility score (0-100)
and an autonomous decision (approve / reject / escalate).

Design invariants (do not break):
  1. Every score is CREDIBILITY (high = trustworthy). Never mix in risk.
  2. An inapplicable check is truly NEUTRAL: it is EXCLUDED from the average,
     so a missing signal neither raises nor lowers credibility. (Averaging in a
     0.5 would drag a strong case toward the middle — that would penalize honest
     photo-only buyers, which we forbid.)
  3. The Defender only ever LIFTS credibility (it argues the claim is legit).
     Its lift is bounded, so it can protect an honest buyer but can NOT rescue a
     case the fraud checks find clearly fraudulent.
  4. No information (no applicable fraud checks) => ESCALATE, never auto-approve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contract import (
    CaseRecord, DECISION_APPROVE, DECISION_REJECT, DECISION_ESCALATE,
    DECISION_RESUBMIT,
)

# The four fraud checks (Defender is handled separately — it points opposite).
FRAUD_CHECKS = ("authenticity", "completeness", "tamper", "relevance")
DEFENDER = "defender"

# Decision thresholds on the 0-100 credibility scale (placeholders; tune vs eval).
APPROVE_AT = 75.0   # >= this  -> auto-approve
REJECT_AT = 35.0    # <  this  -> auto-reject
# in between -> escalate to human

# Max credibility (0-1 scale) the Defender may add on top of the fraud-check base.
# 0.20 = up to 20 points. Enough to save a borderline honest case, not enough to
# lift a clearly fraudulent one (base 0.10 + 0.20 = 0.30 -> still rejects).
DEFENDER_LIFT_MAX = 0.20

# Strong-fraud veto: a single fraud check that is this confident the proof is fake
# is DISPOSITIVE -> auto-reject, and the Defender cannot lift it. This keeps the
# auto-decision rate high on obvious fraud (e.g. clearly AI-edited proof) while
# still letting the Defender protect genuinely borderline honest buyers.
VETO_SCORE = 0.15        # credibility <= this ...
VETO_CONFIDENCE = 0.85   # ... at confidence >= this  => dispositive fraud

# A "low signal" (red flag) is an applicable fraud check scoring below this.
# Convergence guard: with no veto, auto-reject requires >= 2 low signals. A
# single soft red flag can only escalate, never sink a case on its own.
LOW_SIGNAL_SCORE = 0.40
CONVERGENCE_MIN = 2

# Consensus reject: when this many INDEPENDENT fraud checks are red flags, the
# case rejects on the convergence alone, regardless of the weighted mean.
#
# Why the mean needs overriding (measured on case 237722409265023): completeness
# 0.20, tamper 0.20, relevance 0.30 all agreed the video shows a pre-opened box —
# a mean of 0.23 over those three, which rejects. But authenticity scored 0.95 at
# 28% of the weight and pulled the mean to 0.43, over the 35 reject line, so the
# case escalated. Authenticity was not wrong: it answers "were these pixels
# manipulated?", and for a STAGED fraud the honest answer is no — the fraudster
# used a real camera on a genuinely pre-opened parcel. That is exactly the failure
# mode: on staging fraud, an authenticity PASS is guaranteed by construction, and
# averaging it in lets it rescue a case the other three checks convicted.
#
# Averaging assumes the checks are interchangeable votes on one question. They are
# not — they answer different questions, so three of them agreeing is stronger
# evidence than a fourth disagreeing about something else. Hence: count the
# convergence, don't dilute it.
#
# Set to 3 (of 4) deliberately, not 2: CONVERGENCE_MIN=2 still governs the
# score-based path, so two red flags plus a low mean rejects as before, while two
# red flags with a high mean still escalates. Only near-unanimity bypasses the
# score. Measured on the labeled set: 0 false positives, valid_eval unchanged,
# test_eval automation 69% -> 71%.
CONSENSUS_REJECT_MIN = 3

# Irrelevance gate. Relevance at or below the red-flag floor means the proof does
# not depict what is being claimed — so a human reviewer has nothing to adjudicate
# either, and the case bounces for usable proof instead of occupying the queue.
#
# Keyed on relevance ALONE. An earlier version also required both video-only
# checks to be N/A ("photo-only"), which was wrong twice over: it missed cases
# with a video that simply shows the wrong thing, and the "photo-only" half was
# itself a trap — measured, "2 checks N/A -> bounce" hits 100% of the known-valid
# cases and 40% of all traffic, because a PHOTO cannot show an unboxing or a seal.
# Two N/A checks are the signature of a photo submission, not of fraud.
#
# See route() for why this gate runs LAST and why it bounces rather than rejects.
IRRELEVANT_PROOF_MAX = LOW_SIGNAL_SCORE

# Item mismatch: the evidence depicts a DIFFERENT product than the one ordered.
# Treated as its own red flag rather than left to move the relevance score alone,
# because it is a categorically stronger claim than "this proof is unconvincing" —
# damage to something the buyer did not order says nothing about the order, no
# matter how well filmed it is.
#
# Requires CONFIDENCE, not just a low match. The model is explicitly told to
# answer 0.5 when it cannot identify the object, and listings are frequently
# generic, bundled, or multi-pack, so an unsure mismatch is worthless and a
# confident one is decisive. Both bars must be cleared.
ITEM_MISMATCH_MAX = 0.30       # item_match <= this ...
ITEM_MISMATCH_CONFIDENCE = 0.70  # ... at relevance confidence >= this


@dataclass
class ScoreBreakdown:
    """Explainable trace of how the credibility number was reached."""
    base_0_1: float                 # weighted mean of applicable fraud checks
    defender_lift_0_1: float        # bounded lift the Defender contributed
    credibility_0_100: float
    applicable_checks: list[str]
    excluded_checks: list[str]      # inapplicable -> treated as neutral
    no_information: bool            # True => nothing to judge on -> escalate
    vetoed_by: list[str]            # fraud checks that triggered the strong-fraud veto
    hard_reject: bool               # True => dispositive fraud, auto-reject
    low_signals: list[str]          # applicable fraud checks scoring as red flags
    irrelevant_proof: bool          # relevance at/below the red-flag floor
    item_mismatch: bool             # evidence confidently shows a DIFFERENT product
    item_seen: Optional[str]        # what the evidence appears to depict


def combine(rec: CaseRecord) -> ScoreBreakdown:
    """Fold the case's signals into a 0-100 credibility score."""
    applicable: list[str] = []
    excluded: list[str] = []
    vetoed_by: list[str] = []
    low_signals: list[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for name in FRAUD_CHECKS:
        sig = rec.signals.get(name)
        if sig is None or not sig.applicable:
            excluded.append(name)
            continue
        applicable.append(name)
        # Weight by the check's own confidence; a floor keeps a zero-confidence
        # but applicable check from vanishing entirely.
        w = max(sig.confidence, 0.05)
        weighted_sum += sig.score * w
        weight_total += w
        # Strong-fraud veto: this one check is confident enough to be dispositive.
        if sig.score <= VETO_SCORE and sig.confidence >= VETO_CONFIDENCE:
            vetoed_by.append(name)
        # Red flag for the convergence guard. INCLUSIVE (<=): a check landing
        # exactly ON the floor is a red flag. An exclusive test let relevance=0.40
        # register as clean, which auto-approved a photo of a thumbs-down gesture.
        if sig.score <= LOW_SIGNAL_SCORE:
            low_signals.append(name)

    # Irrelevant proof: the submission does not depict what is being claimed.
    # Relevance alone decides this — it is the one check that always applies, and
    # conditioning on the video-only checks was wrong: it made the gate miss any
    # case that HAS an unboxing video which simply shows the wrong thing.
    rel = rec.signals.get("relevance")
    irrelevant_proof = (rel is not None and rel.applicable
                        and rel.score <= IRRELEVANT_PROOF_MAX)

    # Item mismatch: a CONFIDENT read that the evidence shows a different product.
    item_mismatch = (
        rel is not None and rel.applicable
        and rel.item_match is not None
        and rel.item_match <= ITEM_MISMATCH_MAX
        and rel.confidence >= ITEM_MISMATCH_CONFIDENCE
    )
    item_seen = rel.item_seen if rel is not None else None

    no_info = weight_total == 0.0
    base = 0.5 if no_info else weighted_sum / weight_total
    hard_reject = bool(vetoed_by)

    # Defender: bounded, one-directional lift. Suppressed entirely under a veto —
    # dispositive fraud cannot be lifted, even by a strong legitimacy argument.
    lift = 0.0
    dfn = rec.signals.get(DEFENDER)
    if dfn is not None and dfn.applicable and not no_info and not hard_reject:
        # Only the part of the Defender's score ABOVE neutral counts, scaled by
        # its confidence. score=1.0, conf=1.0 -> full lift; score<=0.5 -> none.
        strength = max(0.0, dfn.score - 0.5) * 2.0
        lift = DEFENDER_LIFT_MAX * strength * dfn.confidence

    credibility_0_1 = min(1.0, max(0.0, base + lift))
    return ScoreBreakdown(
        base_0_1=base,
        defender_lift_0_1=lift,
        credibility_0_100=round(credibility_0_1 * 100.0, 1),
        applicable_checks=applicable,
        excluded_checks=excluded,
        no_information=no_info,
        vetoed_by=vetoed_by,
        hard_reject=hard_reject,
        low_signals=low_signals,
        irrelevant_proof=irrelevant_proof,
        item_mismatch=item_mismatch,
        item_seen=item_seen,
    )


def route(breakdown: ScoreBreakdown) -> str:
    """Map a credibility breakdown to an autonomous decision.

    Cost asymmetry: when in doubt, escalate — never auto-approve. No information
    always escalates, regardless of the (neutral) score.

    Anti-over-rejection guard (rubric): auto-reject requires EITHER the strong-
    fraud veto OR multiple converging low signals. A single soft red flag can
    only escalate — it must not sink a case on its own.

    Consensus reject: >= CONSENSUS_REJECT_MIN red flags rejects on convergence
    alone, bypassing the credibility score. The weighted mean can be rescued by
    one unrelated high check (a staged-fraud video is genuinely unmanipulated, so
    Authenticity honestly scores it ~0.95); counting agreeing checks is not
    vulnerable to that, because the checks answer different questions.

    Conflict guard (cost asymmetry): a high MEAN credibility can still hide one
    fraud check screaming red — e.g. a video that looks like a clean unboxing but
    whose tamper/relevance check flags it. We must never AUTO-APPROVE while any
    applicable fraud check is a red flag (< LOW_SIGNAL_SCORE). "Signals conflict
    -> escalate to a human" made mechanical. It only blocks auto-approval; it
    never rejects on its own (that still needs the veto or convergence).
    """
    if breakdown.hard_reject:
        return DECISION_REJECT
    if breakdown.no_information:
        return DECISION_ESCALATE
    # Consensus reject: near-unanimous red flags are dispositive on their own.
    # Checked BEFORE the score gates precisely because the score is the thing
    # that fails here — one unrelated high check can lift the mean above the
    # reject line while three checks agree the proof is staged.
    # A confident item mismatch counts as an additional converging red flag: the
    # proof being of the wrong product is independent evidence from whatever the
    # other checks found, so it should be able to complete a consensus.
    effective_flags = len(breakdown.low_signals) + (1 if breakdown.item_mismatch else 0)
    if effective_flags >= CONSENSUS_REJECT_MIN:
        return DECISION_REJECT

    score = breakdown.credibility_0_100
    if score >= APPROVE_AT:
        # Conflict guard: a lone red-flag fraud check blocks auto-approval.
        decision = DECISION_ESCALATE if breakdown.low_signals else DECISION_APPROVE
    elif score < REJECT_AT:
        # Convergence required: without a veto, a lone red flag escalates.
        decision = (DECISION_REJECT if len(breakdown.low_signals) >= CONVERGENCE_MIN
                    else DECISION_ESCALATE)
    else:
        decision = DECISION_ESCALATE

    # Irrelevance gate — applied LAST, and only to a case already headed for the
    # queue. If the proof doesn't depict the claim, a reviewer has nothing to
    # adjudicate either: they are being handed a photo of a thumbs-down gesture
    # and asked to rule on a refund. Bounce it for usable proof instead.
    #
    # Ordering is the whole design here. Every reject and approve path above is
    # evaluated FIRST and left untouched, so the gate can only ever convert an
    # ESCALATE into a RESUBMIT. Placed any earlier it preempts real fraud: a
    # known AI-edited case (train_eval 237776404294061, relevance 0.20) rejects
    # on the score path, and an earlier gate downgraded it to a mere bounce.
    #
    # It bounces rather than rejects because low relevance does NOT mean fraud.
    # VALID_moldy_food is a genuine claim scoring relevance 0.20: the buyer
    # photographed real mould but selected "Spilled Contents" from the dropdown,
    # so the proof mismatches the STATED REASON while the product and defect are
    # plainly visible. Auto-rejecting on relevance turns that honest buyer into a
    # false positive — measured: 1 FP across the 3 known-valid cases, vs 0 when
    # bouncing. Same operational saving either way; only the buyer pays.
    if decision == DECISION_ESCALATE and breakdown.irrelevant_proof:
        return DECISION_RESUBMIT
    return decision


def score_case(rec: CaseRecord) -> ScoreBreakdown:
    """Combine + route + write the results back onto the CaseRecord."""
    breakdown = combine(rec)
    rec.credibility_score = breakdown.credibility_0_100
    rec.decision = route(breakdown)
    return breakdown
