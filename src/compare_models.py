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
    expected_by_image: dict[str, list[str]] | None = None,
    model_ids: dict[str, str] | None = None,
    image_names: list[str] | None = None,
) -> pd.DataFrame:
    """`image_names`, if given, must be the same length as `images` and is used both as
    the "image" column and as the `expected_by_image` lookup key -- keying by identity
    rather than position, so results stay correct if callers ever reorder `images`.
    """
    model_ids = model_ids or MODELS
    expected_by_image = expected_by_image or {}
    if image_names is not None and len(image_names) != len(images):
        raise ValueError(f"image_names has {len(image_names)} entries, images has {len(images)}.")
    names = image_names if image_names is not None else [str(i) for i in range(1, len(images) + 1)]

    rows = []
    for model_key, model_id in model_ids.items():
        for name, img in zip(names, images):
            df = run_zero_shot(img, labels, model_id=model_id)
            metrics = confidence_metrics(df)
            top1_label = df.loc[0, "label"]
            expected = expected_by_image.get(name, [])
            rows.append(
                {
                    "model": model_key,
                    "image": name,
                    "top1_label": top1_label,
                    "top1_score": round(metrics["top1"], 4),
                    "gap": round(metrics["gap"], 4),
                    "entropy": round(metrics["entropy"], 4),
                    "correct": (top1_label in expected) if expected else None,
                }
            )
    return pd.DataFrame(rows)
