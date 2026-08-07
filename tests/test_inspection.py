from __future__ import annotations

import pytest

from yolo_dashboard.inspection import inspect_detections, parse_inspection_rule
from yolo_dashboard.yolo_inference import Detection


def _detection(label: str) -> Detection:
    return Detection(
        label=label,
        class_id=0,
        confidence=0.9,
        x1=0.0,
        y1=0.0,
        x2=10.0,
        y2=10.0,
    )


def test_parse_inspection_rule_reads_pairs() -> None:
    assert parse_inspection_rule(" nut_ok=4, nut_missing=0 ") == {
        "nut_ok": 4,
        "nut_missing": 0,
    }


def test_parse_inspection_rule_rejects_bad_entry() -> None:
    with pytest.raises(ValueError):
        parse_inspection_rule("nut_ok")
    with pytest.raises(ValueError):
        parse_inspection_rule("nut_ok=dua")
    with pytest.raises(ValueError):
        parse_inspection_rule("nut_ok=-1")


def test_inspect_detections_ok_when_counts_match() -> None:
    detections = [_detection("nut_ok") for _ in range(4)]
    result = inspect_detections(detections, {"nut_ok": 4, "nut_missing": 0})

    assert result.ok is True
    assert result.issues == []
    assert result.counts == {"nut_ok": 4}


def test_inspect_detections_ng_when_nut_missing() -> None:
    detections = [_detection("nut_ok") for _ in range(3)] + [_detection("nut_missing")]
    result = inspect_detections(detections, {"nut_ok": 4, "nut_missing": 0})

    assert result.ok is False
    assert result.issues == ["nut_ok: 3 (harus 4)", "nut_missing: 1 (harus 0)"]


def test_inspect_detections_ignores_label_case() -> None:
    result = inspect_detections([_detection("Nut_OK")], {"nut_ok": 1})

    assert result.ok is True


def test_inspect_detections_ng_on_empty_frame() -> None:
    result = inspect_detections([], {"nut_ok": 4})

    assert result.ok is False
    assert result.counts == {}
