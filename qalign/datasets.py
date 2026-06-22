"""
Dataset adapters: turn a labelled source into Q-Align manifests.

One ``DatasetEntry`` (see qalign.config) describes a source — its format, where
the media lives, and which columns hold the path / score / pre-written answer.
This module reads it and writes two kinds of manifest:

  * **training manifest** (image datasets) — swift conversation records
    (qalign.template.make_record). The assistant target is either a pre-written
    Q-Align answer (``answer_col`` / qalign_json) used verbatim, or a level word
    obtained by binning ``score_col`` through the level scheme.
  * **eval manifest** — scored records ``{modality, src, prompt, stem, gt_score}``
    consumed by qalign.scorer (works for both image and video; video frames are
    sampled at score time, so video eval needs no pre-extraction).

Three formats cover the field: ``csv`` and ``jsonl`` (your own data — give the
column/key names) and ``qalign_json`` (the released Q-Align SFT JSONs, answers
used verbatim). Video *training* manifests are produced by qalign.frames (frame
extraction); this module handles all image manifests and all eval manifests.

Pure Python — no torch / no swift.
"""
import os, csv, json, sys

from .levels import LevelScheme
from .prompts import TaskPrompts
from .template import make_record, strip_image_tokens, write_manifest


# --------------------------------------------------------------------------- #
# Media resolution (with a per-directory listing cache for FUSE-mounted data)
# --------------------------------------------------------------------------- #
_DIR_CACHE = {}


def media_exists(path: str) -> bool:
    """Existence check that lists each parent dir ONCE (one round-trip/dir).

    Over a network/FUSE mount a per-file ``os.path.exists`` is a round-trip;
    hundreds of thousands of them (e.g. AVA) take many minutes. Listing the
    parent directory once and checking membership collapses that cost.
    """
    d = os.path.dirname(path)
    names = _DIR_CACHE.get(d)
    if names is None:
        try:
            names = set(os.listdir(d))
        except OSError:
            names = set()
        _DIR_CACHE[d] = names
    return os.path.basename(path) in names


def resolve_media(entry, rel: str) -> str:
    rel = rel.strip()
    if entry.basename_only:
        return os.path.join(entry.media_root, os.path.basename(rel))
    if entry.media_root:
        return os.path.join(entry.media_root, rel)
    return rel


# --------------------------------------------------------------------------- #
# Source readers -> normalized rows {rel, score, answer}
# --------------------------------------------------------------------------- #
def _rows_csv(entry):
    for r in csv.DictReader(open(entry.source)):
        yield {
            "rel": r[entry.path_col],
            "score": _maybe_float(r.get(entry.score_col)),
            "answer": r.get(entry.answer_col) if entry.answer_col else None,
            "prompt": None,
        }


def _rows_jsonl(entry):
    for line in open(entry.source):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        yield {
            "rel": r[entry.path_col],
            "score": _maybe_float(r.get(entry.score_col)),
            "answer": r.get(entry.answer_col) if entry.answer_col else None,
            "prompt": None,
        }


def _rows_qalign_json(entry):
    """Released Q-Align SFT/test JSON: conversations with a verbatim gpt answer.

    The MOS may live in ``score_col`` (e.g. 'gt_score') or be encoded in the
    ``id`` string as '<file>-><score>' (the KADID convention) — both handled. The
    human turn (which already carries the task phrasing) is kept and reused verbatim.
    """
    data = json.load(open(entry.source))
    for r in data:
        gpt = human = None
        for c in r.get("conversations", []):
            if c.get("from") == "gpt":
                gpt = c["value"]
            elif c.get("from") == "human":
                human = c["value"]
        score = _maybe_float(r.get(entry.score_col))
        if score is None and isinstance(r.get("id"), str) and "->" in r["id"]:
            score = _maybe_float(r["id"].rsplit("->", 1)[-1])
        yield {"rel": r["image"], "score": score, "answer": gpt, "prompt": human}


_READERS = {"csv": _rows_csv, "jsonl": _rows_jsonl, "qalign_json": _rows_qalign_json}


def iter_rows(entry):
    if entry.format not in _READERS:
        raise ValueError(f"dataset '{entry.name}': unknown format '{entry.format}' "
                         f"(known: {list(_READERS)})")
    return _READERS[entry.format](entry)


def _maybe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _score_range(entry, rows):
    """Resolve the [lo, hi] binning range: explicit cfg values or data min/max."""
    lo, hi = entry.score_min, entry.score_max
    if lo is None or hi is None:
        scores = [r["score"] for r in rows if r["score"] is not None]
        if not scores:
            raise ValueError(f"dataset '{entry.name}': no numeric scores found in "
                             f"column '{entry.score_col}' (needed to bin into levels)")
        lo = min(scores) if lo is None else lo
        hi = max(scores) if hi is None else hi
    return lo, hi


# --------------------------------------------------------------------------- #
# Training manifest (image datasets)
# --------------------------------------------------------------------------- #
def build_train_manifest(cfg, name: str, limit: int = 0) -> dict:
    """Build the conversation training manifest for an IMAGE dataset.

    Returns stats {kept, missing, out}. (Video datasets are built by
    qalign.frames; calling this on a video entry raises.)
    """
    entry = cfg.dataset(name)
    if entry.modality != "image":
        raise ValueError(f"dataset '{name}' is modality={entry.modality}; "
                         "video training manifests are built by `qalign frames`")
    scheme = LevelScheme.from_cfg(cfg.levels)
    tp = TaskPrompts.for_task(entry.task, cfg.prompts)

    rows = list(iter_rows(entry))
    need_levels = not entry.answer_col and entry.format != "qalign_json"
    lo = hi = None
    if need_levels:
        lo, hi = _score_range(entry, rows)

    import random
    rng = random.Random(0)
    out = cfg.manifest_path(name)
    kept = missing = 0
    records = []
    for r in rows:
        media = resolve_media(entry, r["rel"])
        if not media_exists(media):
            missing += 1
            continue
        if r["answer"]:                                   # pre-written answer, verbatim
            answer = strip_image_tokens(r["answer"]) if entry.format == "qalign_json" else r["answer"]
            prompt = strip_image_tokens(r["prompt"]) if r.get("prompt") else rng.choice(tp.prompts)
        else:                                             # MOS -> level word
            if r["score"] is None:
                missing += 1
                continue
            level = scheme.map_score(r["score"], lo, hi, dmos=entry.dmos)
            answer = tp.answer(level)
            prompt = rng.choice(tp.prompts)
        records.append(make_record([media], prompt, answer))
        kept += 1
        if limit and kept >= limit:
            break

    n = write_manifest(records, out)
    print(f"[build:{name:>12}] kept={kept} missing={missing} -> {out} ({n} records)",
          file=sys.stderr)
    return {"kept": kept, "missing": missing, "out": out}


# --------------------------------------------------------------------------- #
# Eval manifest (image or video)
# --------------------------------------------------------------------------- #
def build_eval_manifest(cfg, name: str, limit: int = 0) -> dict:
    """Build a scored eval manifest {modality, src, prompt, stem, gt_score}.

    For DMOS sets (higher == worse) the gt is reflected to (lo+hi)-gt so SRCC/PLCC
    come out positive and directly comparable (an order-reversing affine map leaves
    |SRCC| unchanged). Works for image and video; video frames are sampled by the
    scorer at eval time.
    """
    entry = cfg.dataset(name)
    tp = TaskPrompts.for_task(entry.task, cfg.prompts)
    prompt = tp.prompts[0]                                # deterministic eval phrasing
    stem = tp.stem

    rows = list(iter_rows(entry))
    lo = hi = None
    if entry.dmos:
        lo, hi = _score_range(entry, rows)

    out = cfg.eval_manifest_path(name)
    kept = missing = 0
    records = []
    for r in rows:
        if r["score"] is None:
            missing += 1
            continue
        media = resolve_media(entry, r["rel"])
        # video is scored from the source file directly (frames sampled at eval time)
        if entry.modality == "image" and not media_exists(media):
            missing += 1
            continue
        gt = r["score"]
        if entry.dmos:
            gt = (lo + hi) - gt
        records.append({"modality": entry.modality, "src": media,
                        "prompt": prompt, "stem": stem, "gt_score": gt})
        kept += 1
        if limit and kept >= limit:
            break

    n = write_manifest(records, out)
    print(f"[eval-build:{name:>10}] kept={kept} missing={missing} dmos={entry.dmos} "
          f"-> {out} ({n} records)", file=sys.stderr)
    return {"kept": kept, "missing": missing, "out": out}
