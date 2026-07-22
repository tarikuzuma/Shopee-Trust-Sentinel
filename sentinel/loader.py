"""
The ONE loader: maps the organizers' Excel format into CaseRecords.

When real Shopee data arrives in a different shape, this is the only file that
changes. Everything downstream speaks CaseRecord.

Real-data facts this handles:
  - A case = one Order ID, which may span SEVERAL items (extra rows with a
    blank Order ID are continuation rows listing more items for the order).
  - An order may have SEVERAL evidence files (image and/or video).
  - Training sheet has 'Valid / Invalid' + 'Reason for Invalidity'; test sheet
    does not. We map those to true_label when present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import openpyxl

from .contract import CaseRecord, Evidence, ItemListing

# Header labels as they appear in the sheet (row of column names).
COL = {
    "order_id": "Order ID",
    "shop_id": "Shop ID",
    "item_id": "Item ID",
    "listing": "Listing Link",
    "reason": "Return Reason",
    "evidence": "Image/Video Link",
    "validity": "Valid / Invalid",
    "invalidity": "Reason for Invalidity",
}


def _clean(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    # Order/Item IDs come through as floats (e.g. 2.37e14) — normalize to int-string.
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s or None


def _find_header(rows: list[tuple]) -> tuple[int, dict[str, int]]:
    """Locate the header row and return {logical_name: column_index}."""
    for r_idx, row in enumerate(rows):
        cells = [str(c).strip() if c is not None else "" for c in row]
        if COL["order_id"] in cells and COL["reason"] in cells:
            mapping = {}
            for logical, label in COL.items():
                if label in cells:
                    mapping[logical] = cells.index(label)
            return r_idx, mapping
    raise ValueError("Could not locate a header row (no 'Order ID' + 'Return Reason').")


def load_sheet(path: Path | str, sheet_name: str,
               session_id: str) -> list[CaseRecord]:
    """Load one sheet into aggregated CaseRecords (one per Order ID)."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    h_idx, cmap = _find_header(rows)

    def cell(row, key):
        idx = cmap.get(key)
        return _clean(row[idx]) if idx is not None and idx < len(row) else None

    cases: dict[str, CaseRecord] = {}
    current: Optional[CaseRecord] = None

    for row in rows[h_idx + 1:]:
        if all(c is None for c in row):
            continue
        order_id = cell(row, "order_id")

        if order_id:  # start / switch to this order
            if order_id not in cases:
                cases[order_id] = CaseRecord(
                    case_id=order_id,
                    session_id=session_id,
                    return_reason=cell(row, "reason") or "",
                )
            current = cases[order_id]
        # else: continuation row (blank Order ID) — belongs to `current`

        if current is None:
            continue

        # item listing (present on most rows)
        shop_id, item_id = cell(row, "shop_id"), cell(row, "item_id")
        if item_id:
            listing = cell(row, "listing")
            if not any(it.item_id == item_id for it in current.items):
                current.items.append(ItemListing(shop_id or "", item_id, listing))

        # evidence file (may repeat per order)
        ev = cell(row, "evidence")
        if ev and not any(e.filename == ev for e in current.evidence):
            current.evidence.append(Evidence(ev, Evidence.infer_kind(ev)))

        # ground truth, if the sheet carries it (training sheet only)
        validity = cell(row, "validity")
        if validity and current.true_label is None:
            # Store the fine-grained invalidity reason when present, else the label.
            invalid_reason = cell(row, "invalidity")
            current.true_label = (
                "valid" if validity.lower().startswith("valid")
                else (invalid_reason or "invalid")
            )

    return list(cases.values())


if __name__ == "__main__":
    import sys
    xlsx = sys.argv[1] if len(sys.argv) > 1 else None
    if not xlsx:
        print("usage: python -m sentinel.loader <path-to-xlsx> [sheet]")
        raise SystemExit(1)
    sheet = sys.argv[2] if len(sys.argv) > 2 else "Test Data"
    recs = load_sheet(xlsx, sheet, session_id="cli")
    print(f"Loaded {len(recs)} cases from '{sheet}'")
    for r in recs[:12]:
        print(f"  {r.case_id}  reason={r.return_reason!r:20}  "
              f"items={len(r.items)} evidence={[e.filename for e in r.evidence]} "
              f"label={r.true_label}")
