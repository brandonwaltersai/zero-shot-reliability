# Results

CLIP zero-shot image classification, CPU inference, six public stock
images spanning distinct categories (cat, dog, airliner, cab, pizza,
African elephant — one canonical image per class from
[imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images)
plus two common CLIP-demo photos). Reproducible via
[`src/experiments.py`](../src/experiments.py) — every number below traces
to that file, not an ad hoc run.

## 1) Model comparison — six images, twelve coarse labels

| Model | Image | Top-1 | Score | Correct? |
|---|---|---|---:|---|
| clip-vit-base-patch32 | cat | cat | 0.704 | ✓ |
| clip-vit-base-patch32 | dog | dog | 0.822 | ✓ |
| clip-vit-base-patch32 | airliner | airplane | 0.863 | ✓ |
| clip-vit-base-patch32 | cab | car | 0.750 | ✓ |
| clip-vit-base-patch32 | pizza | pizza | 0.982 | ✓ |
| clip-vit-base-patch32 | elephant | elephant | 0.960 | ✓ |
| clip-vit-base-patch16 | cat | cat | 0.660 | ✓ |
| clip-vit-base-patch16 | dog | dog | 0.969 | ✓ |
| clip-vit-base-patch16 | airliner | airplane | 0.977 | ✓ |
| clip-vit-base-patch16 | cab | **vehicle** | 0.529 | ✗ (exact-match) |
| clip-vit-base-patch16 | pizza | pizza | 0.978 | ✓ |
| clip-vit-base-patch16 | elephant | elephant | 0.922 | ✓ |

patch32 gets 6/6 exact string matches; patch16 gets 5/6 — but its one
"miss" is calling the cab photo "vehicle" instead of "car," which isn't
really wrong, it's a different valid label for the same object. That's
itself a finding: measuring "accuracy" against one expected string
undercounts a model that's giving a defensible answer at a different
level of the label hierarchy. Model size doesn't cleanly predict which
model is more confident on a given image either — patch16 is far more
confident on the dog (0.969 vs. 0.822) but less confident on the cat
(0.660 vs. 0.704).

## 2) Label sensitivity — same image, same model, coarse vs. near-miss labels

`clip-vit-base-patch32`, coarse label set (12 broad categories) vs. ten
overlapping near-miss labels specific to that subject (e.g. cat breeds
for the cat photo):

| Image | Coarse top1 / gap / entropy | Near-miss top1 / gap / entropy |
|---|---|---|
| cat | 0.704 / 0.579 / 1.01 | 0.521 / 0.310 / 1.21 |
| dog | 0.822 / 0.692 / 0.64 | 0.582 / 0.289 / 1.15 |
| elephant | 0.960 / 0.923 / 0.18 | 0.926 / 0.895 / 0.39 |

The effect replicates on 2 of 3 subjects but is far weaker on the third:
cat and dog both lose ~0.24–0.28 points of confidence and roughly double
their entropy when the label set gets more granular. The elephant barely
moves (0.960 → 0.926) — the near-miss elephant labels (bull elephant,
tusker, savanna elephant, etc.) are visually harder to tell apart from
each other in general, but this particular photo is apparently still
unambiguous enough that CLIP doesn't get confused by them. **The
label-sensitivity effect is real but not universal — its size depends on
how genuinely visually ambiguous the near-miss labels are for the
specific image, not a fixed property of "adding more labels."**

## 3) Confidently wrong: label set missing the true class

Same three images, one label set that omits the correct answer entirely
(`car, airplane, building, mountain, ocean, furniture`) — a stand-in for
an incomplete label taxonomy, a realistic real-world failure mode:

| Image | Top-1 (wrong) | Score | Guardrail catches it? |
|---|---|---:|---|
| cat | car | 0.509 | **No** |
| dog | airplane | **0.805** | **No** |
| elephant | furniture | 0.414 | Yes |

The default guardrail (`top1 < 0.50` or `gap < 0.10`) catches the
elephant case but misses both the cat and — more strikingly — the dog,
where the model returns "airplane" for a photo of a dog at **80.5%
confidence**, comfortably clearing the threshold. Two out of three
confidently-wrong answers from an incomplete label set pass a
confidence-based safety check undetected. This isn't a threshold-tuning
problem (a higher `top1_threshold` would eventually catch the dog case
too, at the cost of also flagging plenty of correct high-confidence
answers) — it's a category of failure that confidence alone cannot
distinguish from genuine certainty, because from the model's perspective
there genuinely is no better answer inside the label set it was given.

## Why this pairing

Same three-part structure as the other benchmarks in this portfolio:
hold everything fixed except one variable, and let the real numbers show
where the convenient assumption ("high confidence = trustworthy," "more
labels always means more confusion") breaks — and, honestly, where it
doesn't (the elephant case in experiment 2).
