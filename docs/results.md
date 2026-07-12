# Results

CLIP (`openai/clip-vit-base-patch32` and `-patch16`) zero-shot image
classification, CPU inference, on public stock images. Three real,
freshly-run experiments, isolating one variable at a time.

## 1) Model comparison — well-separated labels

Labels: `cat, dog, tiger, lion, golden retriever, tabby cat, house pet,
wild animal, vehicle, food`.

| Model | Image | Top-1 | Score | Gap | Entropy | Correct |
|---|---|---|---:|---:|---:|---|
| clip-vit-base-patch32 | cat photo | tabby cat | 0.943 | 0.903 | 0.274 | ✓ |
| clip-vit-base-patch32 | dog photo | dog | 0.757 | 0.638 | 0.838 | ✓ |
| clip-vit-base-patch16 | cat photo | tabby cat | 0.920 | 0.868 | 0.358 | ✓ |
| clip-vit-base-patch16 | dog photo | dog | 0.959 | 0.943 | 0.233 | ✓ |

Both model sizes get both images right, with genuinely different confidence
profiles per image/model — patch16 is more confident on the dog, patch32
more confident on the cat. Model size alone doesn't predict which one is
more confident on a given input.

## 2) Label sensitivity — same image, same model, different label sets

Same cat photo, `clip-vit-base-patch32`, three label sets:

| Label set | Top-1 | Score | Gap | Entropy |
|---|---|---:|---:|---:|
| Coarse, well-separated (10 labels above) | tabby cat | 0.943 | 0.903 | 0.274 |
| Near-miss (10 overlapping cat-breed terms) | feral cat | 0.521 | 0.310 | 1.210 |
| Mismatched (true class absent from label set) | car | 0.509 | 0.253 | 1.302 |

**Same photo, same model, same underlying pixels.** Swapping the coarse
label set for ten overlapping cat-breed terms drops top-1 confidence from
0.94 to 0.52 and quadruples entropy — the model isn't less capable, the
question it was asked got harder. And when the label set omits the true
class entirely, the model still returns a confident-looking answer
("car", 0.509) with no signal that anything went wrong from the top-1
score alone.

## 3) Why simple confidence thresholds aren't a complete guardrail

The default guardrail here (`flag_low_top1 < 0.50` or `gap < 0.10`) does
**not** fire on the mismatched-label case above (top1=0.509, gap=0.253,
both just above threshold) — a confidently wrong answer, not a low
confidence one. Zero-shot classifiers can look reliable by their own
confidence score while depending entirely on the caller having supplied a
complete label set, which the confidence score cannot verify from inside
the model. Confidence-based flagging catches genuine ambiguity (case 2)
but not label-set incompleteness (case 3) — those need a different check
(e.g., an explicit "none of the above" label, or domain review of the
label taxonomy itself).

## Why this pairing

Same three-part structure as the other benchmarks in this portfolio: hold
everything fixed except one variable, and let the real numbers show where
the convenient assumption (here: "high confidence = trustworthy") breaks.
