"""Zero-shot image classification (CLIP) with confidence-based reliability metrics.

A zero-shot classifier's raw top-1 label is a bad signal to act on directly --
it doesn't tell you when the model is guessing between near-equal options. This
module computes the confidence gap and entropy across the full label distribution,
not just the top prediction, so a caller can flag "this needs a human" cases.
"""
from __future__ import annotations

import math
from functools import lru_cache

import pandas as pd
from PIL import Image
from transformers import pipeline

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=4)
def _get_classifier(model_id: str):
    return pipeline(task="zero-shot-image-classification", model=model_id, device=-1)


def run_zero_shot(image: Image.Image, labels: list[str], model_id: str = DEFAULT_MODEL) -> pd.DataFrame:
    clf = _get_classifier(model_id)
    out = clf(image, candidate_labels=labels)
    df = pd.DataFrame(out).rename(columns={"label": "label", "score": "score"})
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def confidence_metrics(df: pd.DataFrame) -> dict:
    """top1/top2 scores, the gap between them, and Shannon entropy over the full distribution.

    A small gap or high entropy both indicate the model is splitting probability
    across multiple labels rather than confidently picking one.
    """
    scores = df["score"].to_numpy()
    top1 = float(scores[0])
    top2 = float(scores[1]) if len(scores) > 1 else 0.0
    gap = top1 - top2
    entropy = -sum(s * math.log(s + 1e-12) for s in scores if s > 0)
    return {"top1": top1, "top2": top2, "gap": gap, "entropy": entropy}


def flag_low_confidence(metrics: dict, top1_threshold: float = 0.5, gap_threshold: float = 0.1) -> dict:
    """Deployment guardrail: which predictions should escalate to a human reviewer."""
    return {
        "flag_low_top1": metrics["top1"] < top1_threshold,
        "flag_low_gap": metrics["gap"] < gap_threshold,
        "needs_review": metrics["top1"] < top1_threshold or metrics["gap"] < gap_threshold,
    }
