"""
In-training Q-Align eval as an ms-swift TrainerCallback.

Registered into swift's ``callbacks_map`` (import this module via
``--custom_register_path``) and selected with ``--callbacks qalign_eval``. At every
checkpoint save it runs the qalign.scorer SRCC/PLCC eval on the *live* model over
the configured eval sets, logs the curve into ``logging.jsonl`` + per-checkpoint
``eval_metrics.json``, and preserves the top-N checkpoints (by avg SRCC) in
``<output_dir>/best/`` so ``--save_total_limit`` can't rotate the peak away.

Config comes from the YAML pointed to by the ``QALIGN_CFG`` env var (set by
qalign.train), so the callback shares one source of truth with the rest of the run.

Distributed correctness (the rule that bit us once)
---------------------------------------------------
on_save fires on EVERY rank. The forward (the only collective we add) runs in
lockstep on all ranks; reporting is rank-0-only PURE FILE I/O.
  * ZeRO-3: forward through the engine (model_wrapped) so per-layer param
    all-gather collectives have all participants (a rank-0-only forward hangs).
  * We do NOT call ``trainer.log()`` and do NOT call ``dist.barrier()``: both are
    collective/asymmetric, and a rank-0-only invocation desyncs NCCL so the next
    collective deadlocks. ``append_to_jsonl`` is master-only plain file I/O.
Train/eval mode is toggled on ``trainer.model`` and ALWAYS restored in finally.
"""
import os, json

from .config import Config
from .levels import LevelScheme
from . import scorer as QE

try:
    from swift.callbacks.base import TrainerCallback
    from swift.callbacks.mapping import callbacks_map
    from swift.utils import get_logger
    logger = get_logger()
except Exception:                       # importable without swift (won't register)
    TrainerCallback = object
    callbacks_map = {}
    import logging
    logger = logging.getLogger("qalign")


def _load_cfg():
    path = os.environ.get("QALIGN_CFG", "")
    if path and os.path.exists(path):
        return Config.from_yaml(path)
    return Config()


class QAlignEvalCallback(TrainerCallback):
    """Run Q-Align SRCC/PLCC over the eval manifests at every checkpoint save."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.cfg = _load_cfg()
        ev = self.cfg.eval
        self.enabled = ev.in_training
        self.limit = ev.limit
        self.every = max(1, ev.every)
        self.sets = list(ev.sets)
        self.keep_best = ev.keep_best_n > 0
        self.keep_best_n = max(1, ev.keep_best_n)
        self.scheme = LevelScheme.from_cfg(self.cfg.levels)
        self.man_dir = self.cfg.paths.manifest_dir
        self.frame_cache = os.path.join(self.cfg.paths.frames_dir, "eval_cache")
        self._lvl_ids = None
        self._tok = None
        self._n_saves = 0
        self._best_kept = None
        if self.enabled:
            logger.info(f"[qalign-eval-cb] active: sets={self.sets} limit={self.limit} "
                        f"every={self.every} manifests={self.man_dir}")
            if ev.use_ram_cache:
                try:
                    from .cache import install_eval_hook
                    install_eval_hook(self.cfg.paths.eval_cache_dir)
                except Exception as e:
                    logger.warning(f"[qalign-eval-cb] eval RAM cache off "
                                   f"({type(e).__name__}: {e})")
        else:
            logger.info("[qalign-eval-cb] disabled (eval.in_training=false)")

    # --- helpers -----------------------------------------------------------
    def _tokenizer(self):
        if self._tok is None:
            tpl = self.trainer.template
            tok = getattr(tpl, "tokenizer", None)
            if tok is None:
                tok = getattr(self.trainer, "processing_class", None) or \
                      getattr(self.trainer, "tokenizer", None)
            self._tok = tok
        return self._tok

    def _forward_model(self):
        if getattr(self.trainer, "is_deepspeed_enabled", False):
            return self.trainer.model_wrapped          # engine -> ZeRO-3 gather hooks fire
        return self.trainer.model

    def _maybe_keep_best(self, output_dir, ckpt, step, avg):
        """Preserve ckpt in <output_dir>/best/ if avg ranks top-N (hardlink, rotation-proof)."""
        import shutil
        best_dir = os.path.join(output_dir, "best")
        idx_path = os.path.join(best_dir, "best_index.json")
        n = self.keep_best_n
        if self._best_kept is None:
            try:
                self._best_kept = json.load(open(idx_path)).get("kept", [])
            except Exception:
                self._best_kept = []
        kept = [e for e in self._best_kept if e.get("step") != step]
        if len(kept) >= n:
            worst = min(kept, key=lambda e: e["avg"])
            if avg <= worst["avg"]:
                self._best_kept = kept
                return
        dst = os.path.join(best_dir, f"checkpoint-{step}")
        try:
            os.makedirs(best_dir, exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst, ignore_errors=True)
            try:
                shutil.copytree(ckpt, dst, copy_function=os.link)   # cheap + space-free
            except Exception:
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(ckpt, dst)                          # fallback: real copy
            kept.append({"step": step, "avg": avg})
            kept.sort(key=lambda e: e["avg"], reverse=True)
            while len(kept) > n:
                ev = kept.pop()
                shutil.rmtree(os.path.join(best_dir, f"checkpoint-{ev['step']}"),
                              ignore_errors=True)
            keep_names = {f"checkpoint-{e['step']}" for e in kept}
            for name in os.listdir(best_dir):
                p = os.path.join(best_dir, name)
                if name.startswith("checkpoint-") and name not in keep_names and os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
            json.dump({"kept": kept, "best_step": kept[0]["step"], "_avg_srcc": kept[0]["avg"]},
                      open(idx_path, "w"), indent=2)
            json.dump({"best_step": kept[0]["step"], "_avg_srcc": kept[0]["avg"]},
                      open(os.path.join(best_dir, "eval_metrics.json"), "w"), indent=2)
            self._best_kept = kept
            logger.info(f"[qalign-eval-cb] kept top-{n}: "
                        f"{[(e['step'], round(e['avg'], 4)) for e in kept]}")
        except Exception as e:
            logger.warning(f"[qalign-eval-cb] keep-best failed at step {step}: "
                           f"{type(e).__name__}: {e}")

    # --- hook --------------------------------------------------------------
    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        self._n_saves += 1
        if self._n_saves % self.every != 0:
            return
        try:
            self._run(args, state)
        except Exception as e:                          # never kill training on eval error
            logger.warning(f"[qalign-eval-cb] eval skipped at step {state.global_step}: "
                           f"{type(e).__name__}: {e}")

    def _run(self, args, state):
        import numpy as np
        trainer = self.trainer
        template = trainer.template
        tok = self._tokenizer()
        if self._lvl_ids is None:
            self._lvl_ids = QE.level_token_ids(tok, self.scheme.names)
        model = self._forward_model()

        core = getattr(trainer, "model", model)
        was_training = bool(getattr(core, "training", False))
        for m in {id(model): model, id(core): core}.values():
            try:
                m.eval()
            except Exception:
                pass

        results = {}
        try:
            for s in self.sets:
                man = os.path.join(self.man_dir, f"{s}.jsonl")
                if not os.path.exists(man) or os.path.getsize(man) == 0:
                    continue
                # runs on EVERY rank in lockstep -> ZeRO-3 all-gathers stay symmetric
                results[s] = QE.evaluate(
                    model, template, tok, self._lvl_ids, self.scheme.weights, man,
                    limit=self.limit, n_frames=self.cfg.data.frames_per_video,
                    resize_long=self.cfg.data.resize_long, frame_cache_dir=self.frame_cache)
        finally:
            if was_training:
                for m in {id(model): model, id(core): core}.values():
                    try:
                        m.train()
                    except Exception:
                        pass

        # rank-0-only reporting, PURE FILE I/O (no trainer.log, no barrier)
        if state.is_world_process_zero and results:
            srccs = [r["srcc"] for r in results.values() if r.get("srcc") is not None]
            avg = round(float(np.mean(srccs)), 4) if srccs else None
            row = {"step": state.global_step}
            for s, r in results.items():
                if r.get("srcc") is not None:
                    row[f"eval_{s}_srcc"] = r["srcc"]
                if r.get("plcc") is not None:
                    row[f"eval_{s}_plcc"] = r["plcc"]
            if avg is not None:
                row["eval_avg_srcc"] = avg
            try:
                from swift.utils import append_to_jsonl
                append_to_jsonl(os.path.join(args.output_dir, "logging.jsonl"), row)
            except Exception as e:
                logger.warning(f"[qalign-eval-cb] append_to_jsonl failed: {e}")
            logger.info(f"[qalign-eval-cb] step {state.global_step} avg_srcc={avg} :: "
                        + ", ".join(f"{s}={r.get('srcc')}" for s, r in results.items()))
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            out_dir = ckpt if os.path.isdir(ckpt) else args.output_dir
            payload = dict(results)
            payload["_avg_srcc"] = avg
            payload["step"] = state.global_step
            try:
                json.dump(payload, open(os.path.join(out_dir, "eval_metrics.json"), "w"), indent=2)
            except Exception as e:
                logger.warning(f"[qalign-eval-cb] could not write eval_metrics.json: {e}")
            if self.keep_best and avg is not None and os.path.isdir(ckpt):
                self._maybe_keep_best(args.output_dir, ckpt, state.global_step, avg)
        # no dist.barrier(): symmetric forward + rank-0 file I/O -> nothing to resync.


# Register into the SAME dict swift's trainer reads (mutate in place, don't rebind).
if isinstance(callbacks_map, dict):
    callbacks_map["qalign_eval"] = QAlignEvalCallback
    logger.info("[qalign-eval-cb] registered callbacks_map['qalign_eval']")
