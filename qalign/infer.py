"""
Label-free inference: score arbitrary images / videos with a trained checkpoint.

Reuses the qalign.scorer level-token method (no MOS needed). Returns a continuous
score in [0, 1] (higher = better) — the same quantity the SRCC/PLCC eval
correlates against MOS, so it is consistent across media for a given checkpoint
(it is NOT a MOS on a fixed scale; rescale if you need dataset-native units).
"""
import os, sys, json, glob

from .levels import LevelScheme
from .prompts import TaskPrompts

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v"}


def _is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def expand_inputs(patterns):
    """Expand directories / globs / plain paths into a flat ordered file list."""
    out = []
    for p in patterns:
        if os.path.isdir(p):
            out.extend(os.path.join(p, f) for f in sorted(os.listdir(p)))
        elif any(c in p for c in "*?[]"):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    return out


def run(cfg, inputs, model_path=None, task="iqa", as_json=False):
    from .model import load
    from . import scorer as QE

    scheme = LevelScheme.from_cfg(cfg.levels)
    files = expand_inputs(inputs)
    if not files:
        print("no inputs matched", file=sys.stderr); return 1

    model, template, tok = load(cfg, model_path=model_path or cfg.model.path)
    lvl_ids = QE.level_token_ids(tok, scheme.names)
    weights = scheme.weights
    print(f"level token ids: {dict(zip(scheme.names, lvl_ids))}", file=sys.stderr)

    img_tp = TaskPrompts.for_task(task, cfg.prompts)
    vid_tp = TaskPrompts.for_task("vqa", cfg.prompts)
    frame_cache = os.path.join(cfg.paths.frames_dir, "infer_cache")

    if not as_json:
        print(f"{'score':>7}  media")
    for path in files:
        try:
            if _is_video(path):
                tp, modality = vid_tp, "video"
            else:
                tp, modality = img_tp, "image"
            rec = {"modality": modality, "src": path,
                   "prompt": tp.prompts[0], "stem": tp.stem}
            # reuse the exact scorer; also pull per-level probs when asked
            s = QE.score_record(model, template, tok, lvl_ids, weights, rec,
                                n_frames=cfg.data.frames_per_video,
                                resize_long=cfg.data.resize_long,
                                frame_cache_dir=frame_cache)
        except Exception as e:
            print(f"  [warn] {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if as_json:
            print(json.dumps({"media": path, "score": round(s, 6),
                              "task": "vqa" if _is_video(path) else task}, ensure_ascii=False))
        else:
            print(f"{s:7.4f}  {path}")
    return 0
