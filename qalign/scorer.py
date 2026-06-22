"""
Q-Align scorer + SRCC/PLCC evaluation — model-agnostic.

Method (the Q-Align level-token trick):
  build the user turn "<image>...<prompt>", encode it with the model's swift
  template (thinking suppressed if the template supports it), then APPEND the
  answer-stem tokens ("The quality of the image is") so the very next token the
  model predicts is a level word. One forward pass; take the last-position logits,
  restrict to the K level-token ids, softmax, weighted-average with the level
  weights -> a continuous score. Correlate vs MOS with SRCC (spearman) + PLCC
  (pearson).

Nothing here is tied to a specific model: the (model, template, tokenizer) tuple
comes from qalign.model.load(cfg), and the level words/weights from the config.

torch is imported lazily (inside the functions), so this module imports on a box
without a GPU; only ``evaluate`` / ``score_record`` need torch at call time.
"""
import os, sys, json, gc
import numpy as np


# Single indirection for opening a stored image (original src or a cached frame).
# qalign.cache.install_eval_hook monkeypatches this to serve bytes from a
# RAM-resident blob instead of FUSE/GPFS small-file opens.
def open_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def level_token_ids(tokenizer, level_names):
    """First-token id of each (space-prefixed) level word — robust to multi-token words."""
    ids = []
    for w in level_names:
        for cand in (" " + w, w):
            t = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if t:
                ids.append(t[0]); break
        else:
            raise RuntimeError(f"could not tokenize level '{w}'")
    return ids


def sample_frames(path, n_frames=8, resize_long=448, cache_dir=None):
    """Sample N frames from a video; cache them as jpgs so later evals skip decode."""
    paths = None
    if cache_dir:
        stem = path.split(os.sep)[-1]
        stem = stem[:-4] if stem.lower().endswith(".mp4") else stem
        d = os.path.join(cache_dir, stem)
        paths = [os.path.join(d, f"f{i}.jpg") for i in range(n_frames)]
        if all(os.path.exists(p) for p in paths):
            return [open_image(p) for p in paths]
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path, num_threads=1)
    nframes = len(vr)
    idx = [int(round(i)) for i in np.linspace(0, max(nframes - 1, 0), n_frames)]
    batch = vr.get_batch(idx).asnumpy()
    out = []
    from PIL import Image
    if paths:
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    for i in range(n_frames):
        img = Image.fromarray(batch[i]).convert("RGB")
        w, h = img.size
        if max(w, h) > resize_long:
            s = resize_long / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        out.append(img)
        if paths:
            try:
                img.save(paths[i], quality=90)
            except Exception:
                pass
    # decord holds the whole decoded buffer; free it so RAM doesn't accumulate
    # across hundreds of clips (otherwise we get OOM-killed mid-eval).
    del vr, batch
    gc.collect()
    return out


def score_record(model, template, tok, lvl_ids, weights, rec,
                 n_frames=8, resize_long=448, frame_cache_dir=None):
    """Predicted scalar score for one manifest record.

    NOTE: torch.no_grad(), NOT torch.inference_mode(). Under DeepSpeed ZeRO-3,
    params are sharded and gathered on-the-fly during the forward; gathering inside
    inference_mode taints the gathered weight as an "inference tensor" and the next
    training step's backward dies. no_grad gives the same eval speed/memory without
    that taint, so the in-training callback can safely reuse this under ZeRO-3.
    """
    import torch
    with torch.no_grad():
        if rec["modality"] == "image":
            imgs = [open_image(rec["src"])]
        else:
            imgs = sample_frames(rec["src"], n_frames, resize_long, frame_cache_dir)
        content = "<image>" * len(imgs) + rec["prompt"]
        inputs = template.encode({"messages": [{"role": "user", "content": content}],
                                  "images": imgs})
        # append the answer-stem tokens so the model predicts the level word next.
        # (Putting the stem in an assistant turn would shunt it into labels and leave
        # the scored position predicting "The", not a level word.)
        stem_ids = tok(" " + rec["stem"].strip(), add_special_tokens=False)["input_ids"]
        inputs["input_ids"] = list(inputs["input_ids"]) + stem_ids
        if "labels" in inputs:
            inputs["labels"] = list(inputs["labels"]) + [-100] * len(stem_ids)
        batch = template.data_collator([inputs])
        batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        batch.pop("labels", None)
        logits = model(**batch).logits[0, -1]        # position after stem -> next = level word
        sel = logits[lvl_ids].float().softmax(-1).cpu().numpy()
        return float((sel * np.asarray(weights)).sum())


def evaluate(model, template, tok, lvl_ids, weights, manifest, limit=0,
             n_frames=8, resize_long=448, frame_cache_dir=None):
    """Score every record in a manifest and return {n, srcc, plcc}."""
    from scipy.stats import spearmanr, pearsonr
    recs = [json.loads(l) for l in open(manifest)]
    if limit:
        recs = recs[:limit]
    preds, gts = [], []
    for i, r in enumerate(recs):
        try:
            preds.append(score_record(model, template, tok, lvl_ids, weights, r,
                                       n_frames, resize_long, frame_cache_dir))
            gts.append(r["gt_score"])
        except Exception as e:
            print(f"  [warn] {manifest} rec {i}: {type(e).__name__}: {e}", file=sys.stderr)
        if (i + 1) % 200 == 0:
            print(f"  {os.path.basename(manifest)} {i+1}/{len(recs)}", file=sys.stderr)
    if len(preds) < 3:
        return {"n": len(preds), "srcc": None, "plcc": None}
    srcc = float(spearmanr(preds, gts)[0])
    plcc = float(pearsonr(preds, gts)[0])
    return {"n": len(preds), "srcc": round(srcc, 4), "plcc": round(plcc, 4)}


def evaluate_config(cfg, model_path=None, sets=None, limit=None, out=None):
    """High-level eval entrypoint used by the CLI: load model, eval cfg.eval.sets."""
    from .model import load
    from .levels import LevelScheme
    scheme = LevelScheme.from_cfg(cfg.levels)
    sets = sets or cfg.eval.sets
    limit = cfg.eval.limit if limit is None else limit
    mp = model_path or cfg.model.path

    if cfg.eval.use_ram_cache:
        try:
            from .cache import install_eval_hook
            install_eval_hook(cfg.paths.eval_cache_dir)
        except Exception as e:
            print(f"[eval] RAM cache not active ({type(e).__name__}: {e})", file=sys.stderr)

    model, template, tok = load(cfg, model_path=mp)
    lvl_ids = level_token_ids(tok, scheme.names)
    print(f"level token ids: {dict(zip(scheme.names, lvl_ids))}", file=sys.stderr)

    frame_cache = os.path.join(cfg.paths.frames_dir, "eval_cache")
    results = {}
    for s in sets:
        man = cfg.eval_manifest_path(s)
        if not os.path.exists(man) or os.path.getsize(man) == 0:
            print(f"[eval] skip '{s}' (manifest missing/empty: {man})", file=sys.stderr)
            continue
        res = evaluate(model, template, tok, lvl_ids, scheme.weights, man, limit=limit,
                       n_frames=cfg.data.frames_per_video, resize_long=cfg.data.resize_long,
                       frame_cache_dir=frame_cache)
        results[s] = res
        print(json.dumps({s: res}))
    srccs = [r["srcc"] for r in results.values() if r["srcc"] is not None]
    results["_avg_srcc"] = round(float(np.mean(srccs)), 4) if srccs else None
    print(json.dumps({"_avg_srcc": results["_avg_srcc"]}))

    out = out or os.path.join(mp, "eval_metrics.json")
    try:
        json.dump(results, open(out, "w"), indent=2)
        print(f"wrote {out}", file=sys.stderr)
    except Exception as e:
        print(f"[warn] could not write {out}: {e}", file=sys.stderr)
    return results
