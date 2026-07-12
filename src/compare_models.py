"""Compare multiple CLIP checkpoints on the same images/labels -- does model size
actually buy you better label-sensitivity robustness, or just bigger downloads?
"""
from __future__ import annotations

import pandas as pd
from PIL import Image

from .zero_shot import confidence_metrics, run_zero_shot

MODELS = {
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
}


def compare_models(
    images: list[Image.Image],
    labels: list[str],
    expected_by_image: dict[int, list[str]] | None = None,
    model_ids: dict[str, str] | None = None,
) -> pd.DataFrame:
    model_ids = model_ids or MODELS
    expected_by_image = expected_by_image or {}
    rows = []
    for model_key, model_id in model_ids.items():
        for i, img in enumerate(images, start=1):
            df = run_zero_shot(img, labels, model_id=model_id)
            metrics = confidence_metrics(df)
            top1_label = df.loc[0, "label"]
            expected = expected_by_image.get(i, [])
            rows.append(
                {
                    "model": model_key,
                    "image": i,
                    "top1_label": top1_label,
                    "top1_score": round(metrics["top1"], 4),
                    "gap": round(metrics["gap"], 4),
                    "entropy": round(metrics["entropy"], 4),
                    "correct": (top1_label in expected) if expected else None,
                }
            )
    return pd.DataFrame(rows)
