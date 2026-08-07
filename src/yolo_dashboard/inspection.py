from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # ponytail: hindari import cv2 cuma buat type hint
    from yolo_dashboard.yolo_inference import Detection


@dataclass(frozen=True)
class InspectionResult:
    ok: bool
    counts: dict[str, int]
    issues: list[str]


def parse_inspection_rule(raw_value: str) -> dict[str, int]:
    """Parse `nut_ok=4, nut_missing=0` jadi `{"nut_ok": 4, "nut_missing": 0}`."""
    rule: dict[str, int] = {}
    for item in raw_value.split(","):
        entry = item.strip()
        if not entry:
            continue

        label, separator, expected_text = entry.partition("=")
        label = label.strip()
        if not separator or not label:
            raise ValueError(f"Aturan '{entry}' harus berbentuk label=jumlah, contoh: nut_ok=4")

        try:
            expected = int(expected_text.strip())
        except ValueError as error:
            raise ValueError(f"Jumlah pada aturan '{entry}' bukan angka.") from error
        if expected < 0:
            raise ValueError(f"Jumlah pada aturan '{entry}' tidak boleh negatif.")

        rule[label] = expected
    return rule


def inspect_detections(
    detections: list["Detection"],
    rule: dict[str, int],
) -> InspectionResult:
    counts = Counter(detection.label for detection in detections)
    # ponytail: casefold lookup supaya beda kapital di sidebar tidak bikin NG selamanya
    lookup = {label.casefold(): total for label, total in counts.items()}

    issues = [
        f"{label}: {lookup.get(label.casefold(), 0)} (harus {expected})"
        for label, expected in rule.items()
        if lookup.get(label.casefold(), 0) != expected
    ]
    return InspectionResult(ok=not issues, counts=dict(counts), issues=issues)
