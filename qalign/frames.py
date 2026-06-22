"""
Video -> frames, and the video training manifest.

Q-Align scores a video as N uniformly-sampled frames-as-images (default 8). This
both matches the original recipe and sidesteps native-``<video>`` collation bugs
in some swift templates (a single video can render to N vision segments but only
one grid row, raising StopIteration during collation and silently dropping every
video sample).

``build_video_manifest`` reads a *video* DatasetEntry, extracts (and caches) N
frames per clip, and writes a swift conversation manifest of N-frame image lists —
the video counterpart of qalign.datasets.build_train_manifest. It is resumable
(clips whose frames already exist are skipped) and parallel.

decord / PIL are imported lazily inside the worker, so importing this module
costs nothing on a box without them.
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from .datasets import iter_rows, resolve_media
from .levels import LevelScheme
from .prompts import TaskPrompts
from .template import make_record, strip_image_tokens, write_manifest


def _frame_dir(frames_dir: str, media_path: str) -> str:
    stem = os.path.splitext(os.path.basename(media_path))[0]
    # keep one extra parent component to avoid stem collisions across batches
    parent = os.path.basename(os.path.dirname(media_path))
    return os.path.join(frames_dir, parent, stem) if parent else os.path.join(frames_dir, stem)


def _frame_paths(frames_dir, media_path, n):
    d = _frame_dir(frames_dir, media_path)
    return d, [os.path.join(d, f"f{i}.jpg") for i in range(n)]


def _extract_one(args):
    media_path, frames_dir, n, resize_long = args
    d, paths = _frame_paths(frames_dir, media_path, n)
    if all(os.path.exists(p) for p in paths):
        return media_path, paths, "cached"
    if not os.path.exists(media_path):
        return media_path, None, "missing_video"
    try:
        import decord, numpy as np
        from PIL import Image
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(media_path, num_threads=1)
        nframes = len(vr)
        if nframes == 0:
            return media_path, None, "empty"
        idx = [int(round(i)) for i in np.linspace(0, nframes - 1, n)]
        batch = vr.get_batch(idx).asnumpy()
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            img = Image.fromarray(batch[i])
            w, h = img.size
            if max(w, h) > resize_long:
                s = resize_long / max(w, h)
                img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
            img.save(paths[i], quality=90)
        del vr, batch
        return media_path, paths, "extracted"
    except Exception as e:
        return media_path, None, f"error:{type(e).__name__}:{e}"


def build_video_manifest(cfg, name: str, workers: int = 12, limit: int = 0) -> dict:
    """Extract frames for a video dataset and write its training manifest."""
    entry = cfg.dataset(name)
    if entry.modality != "video":
        raise ValueError(f"dataset '{name}' is modality={entry.modality}, not video")
    n = cfg.data.frames_per_video
    resize_long = cfg.data.resize_long
    scheme = LevelScheme.from_cfg(cfg.levels)
    tp = TaskPrompts.for_task(entry.task, cfg.prompts)

    rows = list(iter_rows(entry))
    if limit:
        rows = rows[:limit]
    need_levels = not entry.answer_col and entry.format != "qalign_json"
    lo = hi = None
    if need_levels:
        scores = [r["score"] for r in rows if r["score"] is not None]
        lo = entry.score_min if entry.score_min is not None else (min(scores) if scores else 0.0)
        hi = entry.score_max if entry.score_max is not None else (max(scores) if scores else 1.0)

    # de-dup by media path, remember each clip's prompt+answer
    import random
    rng = random.Random(0)
    meta = {}
    for r in rows:
        media = resolve_media(entry, r["rel"])
        if media in meta:
            continue
        if r["answer"]:
            answer = strip_image_tokens(r["answer"]) if entry.format == "qalign_json" else r["answer"]
            prompt = strip_image_tokens(r["prompt"]) if r.get("prompt") else rng.choice(tp.prompts)
        elif r["score"] is not None:
            answer = tp.answer(scheme.map_score(r["score"], lo, hi, dmos=entry.dmos))
            prompt = rng.choice(tp.prompts)
        else:
            continue
        meta[media] = (prompt, answer)

    print(f"[frames:{name}] clips to process: {len(meta)} (workers={workers})", file=sys.stderr)
    counts = {"cached": 0, "extracted": 0, "missing_video": 0, "empty": 0, "error": 0}
    records, done = [], 0
    tasks = [(m, cfg.paths.frames_dir, n, resize_long) for m in meta]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            media, paths, status = fut.result()
            counts[status if status in counts else "error"] += 1
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)}  {counts}", file=sys.stderr)
            if paths:
                prompt, answer = meta[media]
                records.append(make_record(paths, prompt, answer))

    out = cfg.manifest_path(name)
    nrec = write_manifest(records, out)
    print(f"[frames:{name}] {counts}\n  WROTE {nrec} records -> {out}", file=sys.stderr)
    return {"counts": counts, "records": nrec, "out": out}
