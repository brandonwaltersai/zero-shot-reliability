"""Image loading with validation -- zero-shot classifiers fail silently on bad input."""
from __future__ import annotations

import requests
from PIL import Image
from io import BytesIO

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}


def load_images_from_urls(urls: list[str], timeout: int = 15) -> tuple[list[Image.Image], list[str]]:
    images, sources = [], []
    for url in urls:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        images.append(img)
        sources.append(url)
    return validate_images(images, sources)


def validate_images(
    images: list[Image.Image], sources: list[str] | None = None
) -> tuple[list[Image.Image], list[str]]:
    if not images:
        raise ValueError("No images provided.")
    if sources is None:
        sources = [f"Image {i + 1}" for i in range(len(images))]

    good_images, good_sources = [], []
    for img, src in zip(images, sources):
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w < 2 or h < 2:
            continue
        good_images.append(img)
        good_sources.append(src)

    if not good_images:
        raise ValueError("All images were invalid after validation.")
    return good_images, good_sources


def validate_candidate_labels(labels: list[str], min_labels: int = 2, max_labels: int = 50) -> list[str]:
    cleaned = [str(x).strip() for x in labels if x is not None and str(x).strip()]
    deduped: list[str] = []
    seen = set()
    for x in cleaned:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(x)
    if len(deduped) < min_labels:
        raise ValueError(f"Provide at least {min_labels} non-empty labels, got {len(deduped)}.")
    if len(deduped) > max_labels:
        raise ValueError(f"Too many labels ({len(deduped)}); keep to <= {max_labels}.")
    return deduped
