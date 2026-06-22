"""
The universal template generator.

Turns a (media, prompt, answer) triple into a **model-agnostic** ms-swift
multimodal chat record:

  {"messages": [{"role": "user",      "content": "<image>...<prompt>"},
                {"role": "assistant", "content": "The quality of the image is good."}],
   "images": ["/abs/path", ...]}

This is the only place the Q-Align conversation shape is defined, and it is
deliberately free of any model / template coupling: the record uses swift's
generic ``<image>`` placeholder and a plain path list, so ANY swift VL template
(qwen-vl, internvl, llava, minicpm-v, ...) renders it. Video is represented as a
list of pre-sampled frames-as-images — one ``<image>`` per frame — which matches
the original Q-Align recipe and avoids native-``<video>`` collation pitfalls.

There are two ways the assistant target is produced:
  - **passthrough**: a pre-written Q-Align answer (e.g. "...is excellent.") is used
    verbatim (the faithful path for the released Q-Align SFT JSONs);
  - **leveled**: a level word (from qalign.levels.map_score) is wrapped with the
    task stem via qalign.prompts.TaskPrompts.answer.
"""
import json
from typing import List

IMAGE_TOKEN = "<image>"


def make_record(images: List[str], prompt: str, answer: str) -> dict:
    """Assemble one swift multimodal record.

    Args:
        images: 1 path for an image, N frame paths for a video.
        prompt: the human turn (without any image token; we prepend them here).
        answer: the full assistant target string (e.g. "The quality ... is good.").
    """
    if not images:
        raise ValueError("make_record: at least one image/frame path required")
    user = IMAGE_TOKEN * len(images) + prompt
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "images": list(images),
    }


def strip_image_tokens(text: str) -> str:
    """Drop legacy Q-Align ``<|image|>`` markers — we re-add swift ``<image>`` tokens."""
    return text.replace("<|image|>", "").strip()


def write_manifest(records, path: str) -> int:
    """Write records as one JSON object per line; returns the count."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
