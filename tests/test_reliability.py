import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.images import validate_candidate_labels, validate_images
from src.zero_shot import confidence_metrics, flag_low_confidence
from PIL import Image


def test_validate_candidate_labels_dedupes_case_insensitive():
    labels = validate_candidate_labels(["Cat", "cat", "  Dog  ", "dog"])
    assert labels == ["Cat", "Dog"]


def test_validate_candidate_labels_rejects_too_few():
    with pytest.raises(ValueError):
        validate_candidate_labels(["cat"])


def test_validate_images_drops_tiny_images():
    good = Image.new("RGB", (10, 10))
    tiny = Image.new("RGB", (1, 1))
    images, sources = validate_images([good, tiny], ["good.jpg", "tiny.jpg"])
    assert sources == ["good.jpg"]
    assert len(images) == 1


def test_confidence_metrics_confident_prediction():
    df = pd.DataFrame({"label": ["cat", "dog", "car"], "score": [0.94, 0.04, 0.02]})
    m = confidence_metrics(df)
    assert m["top1"] == pytest.approx(0.94)
    assert m["gap"] == pytest.approx(0.90)
    assert m["entropy"] < 0.5


def test_confidence_metrics_ambiguous_prediction():
    df = pd.DataFrame({"label": ["feral cat", "tabby cat", "stray cat"], "score": [0.35, 0.33, 0.32]})
    m = confidence_metrics(df)
    assert m["gap"] < 0.05
    assert m["entropy"] > 1.0


def test_flag_low_confidence_catches_ambiguous_case():
    metrics = {"top1": 0.35, "top2": 0.33, "gap": 0.02, "entropy": 1.09}
    flags = flag_low_confidence(metrics)
    assert flags["needs_review"] is True


def test_flag_low_confidence_does_not_catch_confidently_wrong_case():
    """The known failure mode this repo documents: a confidently-wrong answer
    from an incomplete label set clears the default threshold."""
    metrics = {"top1": 0.509, "top2": 0.256, "gap": 0.253, "entropy": 1.30}
    flags = flag_low_confidence(metrics)
    assert flags["needs_review"] is False


@pytest.mark.slow
def test_run_zero_shot_end_to_end():
    """Full pipeline on a real image and real CLIP model -- downloads ~600MB, network required."""
    from src.images import load_images_from_urls
    from src.zero_shot import run_zero_shot

    images, _ = load_images_from_urls(
        ["https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/Cat.jpg"]
    )
    df = run_zero_shot(images[0], ["cat", "dog", "car"])
    assert df.iloc[0]["label"] == "cat"
    assert df.iloc[0]["score"] > 0.5


@pytest.mark.slow
def test_mismatched_labels_experiment_reproduces_documented_failure():
    """Regression test for the documented finding: real CLIP (patch32), 3 real images,
    network required. Confirms at least one confidently-wrong case still evades the
    default guardrail -- if this ever stops being true, the README's claim is stale."""
    from src.experiments import run_mismatched_labels

    results = run_mismatched_labels()
    assert set(results.keys()) == {"cat", "dog", "elephant"}
    for r in results.values():
        assert 0.0 <= r["metrics"]["top1"] <= 1.0
    missed_guardrail = [r for r in results.values() if not r["flags"]["needs_review"]]
    assert len(missed_guardrail) >= 1
