# The Q-Align method

Q-Align casts **visual quality assessment as a language task**. Instead of
regressing a number, the model is taught to describe quality with one of K ordered
words, and the score is recovered from the model's probability over those words.

## Levels, not numbers

Human studies collect a **Mean Opinion Score (MOS)** — but humans actually answer
in words ("excellent", "good", "fair", "poor", "bad"). Q-Align trains on those
words. The default vocabulary (best → worst) and weights are:

| level | excellent | good | fair | poor | bad |
|---|---|---|---|---|---|
| weight | 1.0 | 0.75 | 0.5 | 0.25 | 0.0 |

(Configurable — `levels:` in the YAML.)

## Training target

Each training sample becomes a chat turn:

```
user:      <image> How would you rate the quality of this image?
assistant: The quality of the image is good.
```

- The **answer stem** ("The quality of the image is") routes the task. Three stems
  cover the standard tasks — IQA (image quality), IAA (image aesthetics), VQA
  (video quality) — so one model serves all three (this is **ONE-ALIGN**).
- If your dataset ships a continuous MOS, it is binned onto a level by equal-width
  binning over `[min, max]` (`qalign/levels.py`). If it ships pre-written Q-Align
  answers, those are used verbatim.
- **Video** is represented as N uniformly-sampled frames-as-images (default 8),
  one `<image>` token per frame. This matches the original recipe and avoids a
  native-`<video>` collation failure in some templates.

## Scoring (the level-token trick)

At inference the prompt is built to **end exactly at the answer stem**, so the very
next token the model predicts is a level word:

1. encode the user turn `"<image>...<prompt>"`, then append the stem tokens
   (the stem is *not* an assistant turn — that would push it into the labels and
   leave the scored position predicting "The");
2. one forward pass; take the logits at the last position;
3. restrict to the K level-token ids, softmax → a probability per level;
4. weighted average with the level weights → a **continuous score in [0, 1]**.

```
score = Σ_k  softmax(logit_k) · weight_k      # k over {excellent..bad}
```

This score is correlated against MOS with **SRCC** (Spearman, rank) and **PLCC**
(Pearson, linear). Implemented once in `qalign/scorer.py` and reused by the
standalone evaluator, the in-training callback, and inference — so all three
report the same quantity.

## DMOS sets

Some benchmarks (e.g. LIVE, CSIQ) ship **Differential MOS**, where *higher = worse*.
Mark such a dataset `dmos: true`; level mapping inverts, and at eval the ground
truth is reflected to `(min+max) − gt` so SRCC/PLCC come out positive and directly
comparable (an order-reversing affine map leaves |SRCC| unchanged).

## Why it is model-agnostic

Nothing above depends on a particular backbone: the conversation records use
ms-swift's generic `<image>` placeholder, and the scorer only needs a model whose
tokenizer produces the level tokens. Swap `model.path` / `model_type` / `template`
in the YAML to move between Qwen-VL, InternVL, LLaVA, MiniCPM-V, etc.
