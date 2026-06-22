"""
``qalign`` command-line interface.

Subcommands (each takes ``--config CFG`` and optional ``--set key.path=value``
overrides):

  qalign build  --config c.yaml         build image training + eval manifests
  qalign frames --config c.yaml         extract video frames + build video manifests
  qalign cache  --config c.yaml         pack the blob image cache (train / eval)
  qalign train  --config c.yaml [mini]  full-parameter SFT via ms-swift
  qalign eval   --config c.yaml         SRCC/PLCC over the eval sets
  qalign infer  --config c.yaml IMG...  label-free quality scoring of media

The build / frames / cache stages run on CPU; train / eval / infer need the GPU
runtime (ms-swift, torch). See README.md for the end-to-end flow.
"""
import argparse, sys


def _load(args):
    from .config import Config
    return Config.from_yaml(args.config, overrides=args.set or [])


def _add_common(p):
    p.add_argument("--config", required=True, help="experiment YAML")
    p.add_argument("--set", action="append", metavar="key.path=value",
                   help="override a config leaf (repeatable), e.g. --set train.lr=1e-5")


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_build(args):
    from . import datasets as D
    cfg = _load(args)
    # image training manifests (video datasets are handled by `qalign frames`)
    for name in cfg.data.mix:
        entry = cfg.dataset(name)
        if entry.modality == "image":
            D.build_train_manifest(cfg, name, limit=args.limit)
        else:
            print(f"[build] '{name}' is video -> run `qalign frames`", file=sys.stderr)
    # eval manifests (image + video)
    for name in cfg.eval.sets:
        D.build_eval_manifest(cfg, name, limit=args.eval_limit)
    return 0


def cmd_frames(args):
    from . import frames as F
    cfg = _load(args)
    vids = [n for n in cfg.data.mix if cfg.dataset(n).modality == "video"]
    if not vids:
        print("[frames] no video datasets in data.mix", file=sys.stderr)
    for name in vids:
        F.build_video_manifest(cfg, name, workers=args.workers, limit=args.limit)
    return 0


def cmd_cache(args):
    from . import cache as C
    cfg = _load(args)
    if args.which in ("train", "both"):
        mans = [cfg.manifest_path(n) for n in cfg.data.mix]
        C.pack_manifests(mans, cfg.paths.cache_dir, "images.blob", workers=args.workers)
    if args.which in ("eval", "both"):
        mans = [cfg.eval_manifest_path(n) for n in cfg.eval.sets]
        C.pack_manifests(mans, cfg.paths.eval_cache_dir, "eval.blob", workers=args.workers)
    return 0


def cmd_train(args):
    from . import train as T
    cfg = _load(args)
    rc = T.run(cfg, mode=args.mode, dry_run=args.dry_run)
    return rc or 0


def cmd_eval(args):
    from . import scorer as S
    cfg = _load(args)
    S.evaluate_config(cfg, model_path=args.model, sets=args.sets,
                      limit=args.limit, out=args.out)
    return 0


def cmd_infer(args):
    from . import infer as I
    cfg = _load(args)
    return I.run(cfg, args.inputs, model_path=args.model, task=args.task, as_json=args.json)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="qalign",
                                 description="Modernized, model-agnostic Q-Align toolkit.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build image training + eval manifests")
    _add_common(p)
    p.add_argument("--limit", type=int, default=0, help="cap training records/dataset (0=all)")
    p.add_argument("--eval-limit", type=int, default=0, help="cap eval records/set (0=all)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("frames", help="extract video frames + build video manifests")
    _add_common(p)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="cap clips/dataset (0=all)")
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("cache", help="pack the blob image cache")
    _add_common(p)
    p.add_argument("--which", choices=["train", "eval", "both"], default="both")
    p.add_argument("--workers", type=int, default=64)
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("train", help="full-parameter SFT via ms-swift")
    _add_common(p)
    p.add_argument("mode", nargs="?", choices=["full", "mini"], default="full")
    p.add_argument("--dry-run", action="store_true", help="print the swift command and exit")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="SRCC/PLCC over the eval sets")
    _add_common(p)
    p.add_argument("--model", default="", help="checkpoint dir to eval (default model.path)")
    p.add_argument("--sets", nargs="*", default=None, help="subset of eval.sets")
    p.add_argument("--limit", type=int, default=None, help="records/set (default eval.limit)")
    p.add_argument("--out", default="", help="metrics json (default <model>/eval_metrics.json)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("infer", help="label-free quality scoring of media")
    _add_common(p)
    p.add_argument("inputs", nargs="+", help="image/video files, globs, or directories")
    p.add_argument("--model", default="", help="checkpoint dir (default model.path)")
    p.add_argument("--task", choices=["iqa", "iaa", "vqa"], default="iqa")
    p.add_argument("--json", action="store_true", help="emit JSON lines")
    p.set_defaults(func=cmd_infer)

    args = ap.parse_args(argv)
    # normalize empty-string optionals to None where the handler expects it
    if getattr(args, "model", None) == "":
        args.model = None
    if getattr(args, "out", None) == "":
        args.out = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
