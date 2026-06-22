# Adapting qalign

Everything is driven by one YAML. Copy `configs/example_iqa.yaml` and edit. This
page covers the two adaptations you'll actually do: **a new model** and **a new
dataset**.

## Use a different model

Q-Align works on any ms-swift-registered vision-language model. Change three
fields:

```yaml
model:
  path: /path/to/internvl-or-qwen-or-llava   # local dir or HF id
  model_type: internvl                       # the swift model_type
  template: internvl                          # the swift template (often == model_type)
  enable_thinking: false                      # ignored by templates without a thinking mode
  max_length: 8192                            # keep >= 8192 if you train on video
```

That's it — the template generator, cache, scorer, callback, and trainer are all
backbone-independent. Pick `model_type` / `template` from swift's registry
(`swift sft --help`, or swift's model/template docs). If a template has no
"thinking" concept, `enable_thinking` is simply ignored.

For large models, set `train.deepspeed: zero3` (or `zero3_offload`) and raise
`train.gpus`.

## Add your own dataset

A dataset is one entry under `data.datasets`. Two source shapes are supported.

### A) You have media + a MOS column (CSV or JSONL)

```yaml
data:
  datasets:
    - name: my_iqa
      task: iqa                 # iqa | iaa | vqa  -> chooses the prompt + answer stem
      format: csv               # or jsonl
      source: /data/my/labels.csv
      media_root: /data/my/images   # prefixed to each path
      modality: image           # or video
      path_col: filename        # column holding the (relative) media path
      score_col: mos            # column holding the continuous score
      dmos: false               # true if higher score == worse (LIVE/CSIQ-style)
      # score_min / score_max:  fix the binning range (else min/max of the data)
      # basename_only: true     # resolve media by basename under media_root
  mix: [my_iqa]                 # train on it
```

The continuous score is binned into the level words (`docs/METHOD.md`). For a
**video** dataset set `modality: video` and run `qalign frames` (it samples the
frames and writes the manifest); images use `qalign build`.

### B) You have pre-written Q-Align answers (the released SFT JSONs)

```yaml
    - {name: koniq, task: iqa, format: qalign_json, modality: image,
       source: /data/Q-Align/train_koniq.json, media_root: /data/q-align-dataset}
```

`qalign_json` reads the released conversation JSONs and uses the `gpt` answer
verbatim (no MOS→level recompute). For eval, the score is read from `score_col`
(default `gt_score`) or parsed from an `id` like `"file->4.63"`.

### Then build → cache → train → eval

```bash
qalign build  --config configs/my.yaml     # image train manifests + eval manifests
qalign frames --config configs/my.yaml     # (only if you have video datasets)
qalign cache  --config configs/my.yaml     # pack the blob cache (big speedup)
qalign train  --config configs/my.yaml mini   # 10-step smoke test first
qalign train  --config configs/my.yaml         # full run
qalign eval   --config configs/my.yaml --model <checkpoint>
```

## Change the levels or prompts

```yaml
levels:
  names:   [perfect, great, ok, bad, terrible]   # any K, BEST -> WORST
  weights: [1.0, 0.75, 0.5, 0.25, 0.0]

prompts:                          # optional per-task overrides
  iqa:
    stem: "The quality of the image is"
    prompts: ["Rate this image.", "How good is this image?"]
```

The scorer reads the K level tokens from `levels.names`, so the vocabulary, the
count, and the language are all yours.

## Eval sets

Each entry in `eval.sets` must reference a dataset with a **ground-truth score per
record** (a `score_col`, or `gt_score` in a Q-Align test JSON). Define separate
`*_test` entries for your held-out / cross-dataset splits and list them under
`eval.sets`. The in-training callback (`eval.in_training: true`) scores the live
model at every checkpoint and preserves the top-`keep_best_n` checkpoints by avg
SRCC in `<output_dir>/best/`.

## Common knobs

| want to… | set |
|---|---|
| LoRA instead of full FT | `train.train_type: lora` |
| freeze the vision tower | `train.freeze_vit: true` |
| bigger model / OOM | `train.deepspeed: zero3` (or `zero3_offload`), raise `train.gpus` |
| fewer/faster eval records | `eval.limit: 100` |
| override anything once | `--set train.lr=1e-5` on any command |
