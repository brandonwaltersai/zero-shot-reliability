"""Reproduces the experiments documented in docs/results.md.

Run directly (`python -m src.experiments`) to regenerate the numbers --
everything in the results doc traces back to this file, not an ad hoc run.
"""
from __future__ import annotations

from .compare_models import compare_models
from .images import load_images_from_urls, validate_candidate_labels
from .zero_shot import confidence_metrics, flag_low_confidence, run_zero_shot

IMAGE_URLS = {
    "cat": "https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/Cat.jpg",
    "dog": "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "airliner": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02690373_airliner.JPEG",
    "cab": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02930766_cab.JPEG",
    "pizza": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n07873807_pizza.JPEG",
    "elephant": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02504458_African_elephant.JPEG",
}

COARSE_LABELS = [
    "cat", "dog", "airplane", "car", "pizza", "elephant",
    "tiger", "lion", "house pet", "wild animal", "vehicle", "food",
]

EXPECTED_BY_IMAGE = {1: ["cat"], 2: ["dog"], 3: ["airplane"], 4: ["car"], 5: ["pizza"], 6: ["elephant"]}

NEAR_MISS_LABELS = {
    "cat": ["tabby cat", "house cat", "domestic cat", "kitten", "calico cat",
            "orange cat", "stray cat", "feral cat", "siamese cat", "persian cat"],
    "dog": ["golden retriever", "labrador", "puppy", "terrier", "hound",
            "sheepdog", "poodle", "mixed breed dog", "stray dog", "working dog"],
    "elephant": ["african elephant", "asian elephant", "bull elephant", "elephant calf",
                 "elephant herd", "tusker", "young elephant", "wild elephant",
                 "savanna elephant", "forest elephant"],
}

MISMATCHED_LABELS = ["car", "airplane", "building", "mountain", "ocean", "furniture"]


def run_model_comparison():
    """Experiment 1: two CLIP checkpoints, six diverse images, coarse labels."""
    images, _ = load_images_from_urls(list(IMAGE_URLS.values()))
    labels = validate_candidate_labels(COARSE_LABELS)
    return compare_models(images, labels, expected_by_image=EXPECTED_BY_IMAGE)


def run_label_sensitivity():
    """Experiment 2: same image/model, coarse vs. near-miss labels, per subject."""
    results = {}
    for name in NEAR_MISS_LABELS:
        images, _ = load_images_from_urls([IMAGE_URLS[name]])
        img = images[0]
        coarse = confidence_metrics(run_zero_shot(img, COARSE_LABELS))
        near_miss = confidence_metrics(run_zero_shot(img, NEAR_MISS_LABELS[name]))
        results[name] = {"coarse": coarse, "near_miss": near_miss}
    return results


def run_mismatched_labels():
    """Experiment 3: label set that omits the true class entirely, per image."""
    results = {}
    for name in ("cat", "dog", "elephant"):
        images, _ = load_images_from_urls([IMAGE_URLS[name]])
        img = images[0]
        df = run_zero_shot(img, MISMATCHED_LABELS)
        metrics = confidence_metrics(df)
        results[name] = {
            "top1_label": df.iloc[0]["label"],
            "metrics": metrics,
            "flags": flag_low_confidence(metrics),
        }
    return results


if __name__ == "__main__":
    print("=== Model comparison ===")
    print(run_model_comparison().to_string())
    print("\n=== Label sensitivity ===")
    for name, r in run_label_sensitivity().items():
        print(name, r)
    print("\n=== Mismatched labels ===")
    for name, r in run_mismatched_labels().items():
        print(name, r)
