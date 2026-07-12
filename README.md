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

### The finding
Swapping a coarse label set for ten overlapping near-miss labels on the
**same photo, same model** drops top-1 confidence from 0.94 to 0.52 and
quadruples entropy. Worse: when the true class is missing from the label
set entirely, the model still returns a confident-looking wrong answer
(0.509) that clears the default confidence guardrail — meaning
threshold-based flagging catches genuine ambiguity but not an incomplete
label taxonomy, which needs a different check entirely.

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
