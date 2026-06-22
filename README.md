<div align="center">

<img src="assets/logo.png" alt="qalign — Pro (9B) · Lite (4B) · Mini (0.8B)" width="62%">

# Q-ReAlign

**Q-ReAlign: Lightweight Human-Aligned Multimodal Judges Built on Modern Vision-Language Models**

[The method](docs/METHOD.md) · [Adapting guide](docs/ADAPTING.md) · [Quickstart](#5-minute-tour-no-gpu-bundled-toy-data) · [Results](#results)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Q--ReAlign%20Collection-ffcc00)](https://huggingface.co/collections/q-future/q-realign)

</div>

---
*Q-Align-level performance with 50% fewer parameters. Easy to use, easy to train.*
4G Memory. 

## Results

We train three sizes — **Mini (0.8B) · Lite (4B) · Pro (9B)** (the mascots above) —
and compare against the original Q-Align.

<h3 align="center">Performance — SRCC across 7 benchmarks</h3>

<div align="center">
  <img src="assets/Radar.png" alt="SRCC across KonIQ, SPAQ, KADID, AGI, LIVE, AVA, LSVQ" width="68%">
</div>

<p align="center">
All three variants <b>match or beat</b> the original Q-Align across KonIQ, SPAQ,
KADID, AGI, LIVE, AVA, and LSVQ, and quality scales cleanly with model size —
Pro (9B) reaches <b>avg SRCC 0.896 vs. Q-Align's 0.869</b>.
</p>

Per-dataset **SRCC / PLCC** on seven QA benchmarks — **bold** = best, *italic* = second-best:

| Model | KonIQ | SPAQ | KADID | AGI | LIVE | AVA | LSVQ | **Avg.** |
|---|---|---|---|---|---|---|---|---|
| Q-Align | 0.942 / *0.944* | 0.932 / 0.933 | 0.912 / 0.920 | 0.738 / 0.781 | 0.897 / 0.870 | 0.798 / 0.796 | 0.867 / 0.866 | 0.869 / 0.873 |
| Mini (0.8B) | 0.935 / 0.938 | 0.931 / 0.933 | 0.903 / 0.907 | 0.811 / 0.848 | **0.907** / *0.873* | 0.797 / 0.794 | 0.869 / 0.869 | 0.879 / 0.880 |
| Lite (4B) | *0.943* / 0.941 | *0.932* / *0.934* | *0.928* / *0.931* | *0.829* / *0.871* | 0.899 / 0.862 | *0.814* / *0.804* | *0.880* / *0.879* | *0.889* / *0.889* |
| Pro (9B) | **0.950** / **0.952** | **0.935** / **0.937** | **0.934** / **0.939** | **0.843** / **0.885** | *0.902* / **0.876** | **0.832** / **0.828** | **0.883** / **0.884** | **0.896** / **0.900** |

<sub>Each cell is SRCC / PLCC. Rankings use full-precision values before rounding; avg. SRCC for Pro (9B) and Mini (0.8B) uses the reported 0.8956 / 0.8792. In the original table, every AIGC10K result beating Q-Align is highlighted in red — here, all but a handful of cells qualify.</sub>

<h3 align="center">Speed — throughput vs. batch size</h3>

<div align="center">
  <img src="assets/Speed.png" alt="Throughput (images/sec) on RTX 4090 and H200 141GB" width="100%">
</div>

<p align="center">
Sustained throughput on a consumer <b>RTX 4090</b> and a datacenter <b>H200 141GB</b>, measured on the <b>SPAQ</b> dataset.<br>
Mini (0.8B) tops out at <b>26.7 img/s @ bs=4</b> (4090) and <b>61.1 img/s @ bs=14</b>
(H200); OOM points are marked where the batch size exceeds device memory.
</p>

## Models

Pretrained weights are on the Hugging Face Hub —
[**q-future/Q-ReAlign** collection](https://huggingface.co/collections/q-future/q-realign):

| Model | Params | Hugging Face |
|---|---|---|
| Pro  | 9B   | [`q-future/Q-ReAlign-Pro-9B`](https://huggingface.co/q-future/Q-ReAlign-Pro-9B) |
| Lite | 4B   | [`q-future/Q-ReAlign-Lite-4B`](https://huggingface.co/q-future/Q-ReAlign-Lite-4B) |
| Mini | 0.8B | [`q-future/Q-ReAlign-Mini-0.8B`](https://huggingface.co/q-future/Q-ReAlign-Mini-0.8B) |

Pass any of these repo IDs as `CKPT` / `--model` and the weights download automatically.

## Quick Run
```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
# transformers >= 5.2.0 for Qwen3.5 Support

CKPT, IMAGE = "q-future/Q-ReAlign-Mini-0.8B", "photo.jpg"   # auto-downloads from the Hub
LEVELS, WEIGHTS = ["excellent", "good", "fair", "poor", "bad"], [1.0, 0.75, 0.5, 0.25, 0.0]

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModelForImageTextToText.from_pretrained(CKPT, dtype="auto").to(device).eval()

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How would you rate the quality of this image?"}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True) + "The quality of the image is"
inputs = processor(text=[text], images=[Image.open(IMAGE).convert("RGB")], return_tensors="pt").to(device)

ids = [processor.tokenizer(" " + w, add_special_tokens=False).input_ids[0] for w in LEVELS]
probs = model(**inputs).logits[0, -1, ids].softmax(-1)
score = (probs * torch.tensor(WEIGHTS, device=device)).sum().item()
print(f"quality score: {score:.4f}")   # 0 (worst) .. 1 (best)
```

## Install

```bash
pip install -e .            # core (CPU): template / cache / config layer
pip install -e ".[runtime]" # + ms-swift, transformers, deepspeed, decord (GPU box)
pip install -e ".[dev]"     # + pytest
```

`import qalign` works without a GPU; ms-swift / torch are imported lazily and only
needed for `train` / `eval` / `infer`.

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="26" align="top"> Docker

The fastest way to a working `train / eval / infer` box. The image builds on the
**official, verified ModelScope + ms-swift image** — which already ships torch
2.10, ms-swift 4.0.3, transformers, deepspeed, and vLLM on CUDA 12.8.1 /
Python 3.11 — and just layers `qalign` (+ decord for video) on top:

```bash
# build (qalign on top of the verified runtime base)
docker build -t qalign .

# run with GPUs, mounting the repo + your datasets
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/qalign \
  -v /path/to/datasets:/data \
  qalign

# inside the container the CLI is ready:
qalign build --config configs/example_iqa.yaml
qalign train --config configs/my.yaml mini
```

The base image is the source of truth for the swift runtime, so the build never
touches torch / ms-swift. For a region-local pull, override the base:

```bash
docker build -t qalign \
  --build-arg BASE=modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3 .
```

## 5-minute tour (no GPU, bundled toy data)

The CPU stages run out of the box on a tiny synthetic dataset:

```bash
pip install -e ".[dev]"
pytest -q                                            # unit tests for the core
qalign build --config configs/example_iqa.yaml       # -> examples/runs/{data,eval_manifests}/toy_iqa.jsonl
qalign cache --config configs/example_iqa.yaml --which train   # -> examples/runs/cache/{images.blob,index.json}
```

Open the produced `toy_iqa.jsonl` to see the Q-Align conversation records. To
actually train/score, set `model.path` in the config to a real VL model and run on
a GPU box.

## The one CLI

```bash
qalign build  --config CFG    # labelled sources -> Q-Align training + eval manifests
qalign frames --config CFG    # videos -> N sampled frames-as-images (+ manifest)
qalign cache  --config CFG    # pack images into a RAM-resident blob (big dataloading win)
qalign train  --config CFG    # full-parameter SFT via ms-swift  (append `mini` for a smoke test)
qalign eval   --config CFG    # SRCC / PLCC over the eval sets
qalign infer  --config CFG IMG/VIDEO ...   # label-free quality score per file
```

Every command takes `--config` and optional `--set key.path=value` overrides
(e.g. `--set train.lr=1e-5`).

## End-to-end (real run)

```bash
# 1) edit a config: point model.path at your VL model, list your datasets + mix
cp configs/example_iqa.yaml configs/my.yaml && $EDITOR configs/my.yaml

# 2) prepare data  (build = images + eval manifests; frames = video)
qalign build  --config configs/my.yaml
qalign frames --config configs/my.yaml          # only if you have video datasets
qalign cache  --config configs/my.yaml          # optional but recommended at scale

# 3) train (smoke test first), then full run
qalign train  --config configs/my.yaml mini
qalign train  --config configs/my.yaml

# 4) evaluate a checkpoint  (or rely on the in-training eval curve in logging.jsonl)
qalign eval   --config configs/my.yaml --model runs/.../checkpoint-XXXX

# 5) score arbitrary media
qalign infer  --config configs/my.yaml --model runs/.../best/checkpoint-XXXX photo.jpg clip.mp4
```

`configs/onealign.yaml` is the full multi-task reference (IQA + IAA + VQA = ONE-ALIGN).

### Full-parameter or LoRA

Training defaults to full-parameter SFT. To fine-tune with **LoRA** instead, set
`train.train_type: lora` in the config (or override it once on the command line) —
everything else in the pipeline is unchanged:

```yaml
train:
  train_type: lora          # full (default) | lora
  lora_rank: 16             # LoRA rank
  lora_alpha: 32            # LoRA alpha
  # target_modules default to all-linear; LR defaults to 2e-4 for LoRA (2e-5 full)
```

```bash
# one-off LoRA run without editing the file
qalign train --config configs/my.yaml --set train.train_type=lora
qalign train --config configs/my.yaml --set train.train_type=lora --set train.lora_rank=32
```

ms-swift attaches the adapters to `all-linear` targets and picks the faithful LoRA
learning rate (2e-4) automatically; the eval / infer / cache stages are identical.

## Repository layout

```
qalign/
  config.py    one YAML -> nested dataclasses; --set overrides           (the control surface)
  levels.py    level vocabulary, weights, MOS -> level binning           (the science)
  prompts.py   per-task prompt pools + answer stems (iqa / iaa / vqa)
  template.py  the universal template generator: record -> swift jsonl
  datasets.py  csv / jsonl / qalign_json adapters -> train & eval manifests
  frames.py    video -> N frames-as-images + video training manifest
  cache.py     pack images into a byte-exact blob; mmap RAM-resident load hooks
  model.py     load (model, template, tokenizer) from config            (the only swift coupling)
  scorer.py    level-token softmax score + SRCC/PLCC eval
  callback.py  in-training eval (distributed-safe) + best-checkpoint keeping
  train.py     compose & launch `swift sft` from the config
  infer.py     label-free scoring of arbitrary media
  cli.py       the `qalign` dispatcher
configs/   example_iqa.yaml (minimal)  ·  onealign.yaml (full reference)
docs/      METHOD.md (the method)      ·  ADAPTING.md (new model / new dataset)
examples/  a tiny synthetic dataset so the CPU stages run immediately
tests/     pure-python unit tests (no GPU)
scripts/   train.sh  ·  launch_pod.example.sh
```

## Design notes

- **Model-agnostic by construction.** Records use swift's generic `<image>`
  placeholder; the scorer only needs the level tokens. The backbone lives entirely
  in `model:` of the YAML.
- **Faithful Q-Align defaults**, all overridable: 5 levels with weights
  `[1, .75, .5, .25, 0]`, 8 frames/video, full-parameter FT in bf16, LR 2e-5,
  cosine schedule, warmup 0.03, 2 epochs, vision tower + projector trainable.
- **The dataloading cache** packs every image byte-for-byte into one mmap'd blob
  so training/eval read RAM slices instead of hundreds of thousands of tiny files
  — the difference between a starved GPU and a fed one at QA dataset scale.
- **Robust in-training eval.** The callback runs the *same* scorer on the live
  model at each save, logs the SRCC/PLCC curve, and hard-links the top-N
  checkpoints out of the rotation path so the peak is never lost — and it is
  distributed-correct under DeepSpeed ZeRO-2/3 (no rank-desync hangs).

## Requirements

Python ≥ 3.9. Runtime training/eval needs ms-swift (4.0.2), transformers,
deepspeed, and decord (for video) — see `requirements.txt`. Provided by your
training image or `pip install -e ".[runtime]"`.

## Repo Maintained by
[@Yushuo Zheng](https://github.com/Frank-Y-Zheng) , [@Zicheng Zhang](https://github.com/zzc-1998)

## Acknowledgements

`qalign` stands on the shoulders of three projects — this repository would not
exist without them, and we are grateful to their teams.

- **[Q-Align](https://github.com/Q-Future/Q-Align)** (Q-Future) — the method this
  toolkit modernizes. Q-Align introduced visual scoring via *discrete text-defined
  levels* and unified IQA + IAA + VQA into ONE-ALIGN. Everything in `levels.py`,
  `prompts.py`, and `scorer.py` follows their recipe. Thanks to Haoning Wu and the
  entire Q-Align / ONE-ALIGN team.

- **[ms-swift](https://github.com/modelscope/ms-swift)** (the ModelScope SWIFT
  team) — the training/inference backbone we build on. SWIFT's generic multimodal
  templates and `swift sft` pipeline are what let `qalign` stay model-agnostic: a
  new backbone is a config edit, not a code change. Thanks to the ms-swift team.

- **[Qwen3.5](https://github.com/QwenLM/Qwen3-VL)** (the Qwen team, Alibaba) — the
  reference vision-language backbone (`model_type: qwen3_5`) behind our results.
  Thanks to the Qwen team for releasing such a capable open VL model.

If you use this toolkit, please also cite the original works:

```bibtex
@inproceedings{wu2024qalign,
  title     = {Q-Align: Teaching {LMM}s for Visual Scoring via Discrete Text-Defined Levels},
  author    = {Wu, Haoning and Zhang, Zicheng and Zhang, Weixia and Chen, Chaofeng and
               Liao, Liang and Li, Chunyi and Gao, Yixuan and Wang, Annan and Zhang, Erli and
               Sun, Wenxiu and Yan, Qiong and Min, Xiongkuo and Zhai, Guangtao and Lin, Weisi},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024}
}

@inproceedings{swift2025,
  title     = {{SWIFT}: A Scalable Lightweight Infrastructure for Fine-Tuning},
  author    = {ModelScope Team},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2025},
  note      = {\url{https://github.com/modelscope/ms-swift}}
}

@misc{qwen3_5,
  title        = {Qwen3.5: Towards Native Multimodal Agents},
  author       = {Qwen Team},
  year         = {2025},
  howpublished = {\url{https://github.com/QwenLM/Qwen3-VL}}
}
```
