# Zero-Shot Image Classification Reliability

![tests](https://github.com/brandonwaltersai/zero-shot-reliability/actions/workflows/tests.yml/badge.svg)

CLIP zero-shot image classification is only as reliable as the label set
you hand it. This measures that directly: same image, same model, three
label sets, and a look at where simple confidence-threshold guardrails
fail to catch it.

## What's here

### `src/zero_shot.py` / `src/compare_models.py`
Run CLIP (`transformers` pipeline) on an image against a candidate label
list, and compute confidence-gap and entropy metrics across the *full*
score distribution — not just the top-1 label, which hides how close the
model actually was to picking something else.

### `src/experiments.py`
Three experiments across **six diverse images** and two CLIP checkpoints
(not a single cat photo) — model comparison, label sensitivity, and an
incomplete-label-set failure mode. Every number in `docs/results.md`
traces to this file (`python -m src.experiments` reproduces it exactly).

### The findings
Near-miss labels (e.g. ten overlapping cat-breed terms instead of 12
broad categories) tank confidence and roughly double entropy on 2 of 3
tested subjects — but barely move it on the third, so **the effect size
depends on the image, not a universal law of "more labels = more
confusion."** Separately: when the true class is missing from the label
set entirely, the model still returns confident-looking wrong answers —
one at **80.5% confidence** (a dog photo labeled "airplane") — and the
default confidence guardrail misses 2 of 3 such cases. Threshold-based
flagging catches genuine ambiguity; it does not reliably catch an
incomplete label taxonomy, because there's no signal inside the model's
own confidence score that distinguishes the two failure modes.

**Full experiment writeup: [`docs/results.md`](docs/results.md).**

## Why this pairing

The other benchmarks in this portfolio isolate one variable at a time to
show where a convenient default breaks. Here it's "trust the confidence
score" — true right up until the label set itself is the problem, which
the confidence score has no way to see.

## Stack

Python · transformers (CLIP) · PyTorch (CPU) · pandas

## Author

Brandon Walters — [LinkedIn](https://www.linkedin.com/in/bw172b29208/)
