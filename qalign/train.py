"""
Compose and launch ``swift sft`` from a config.

This is the model-agnostic, config-driven replacement for the old hand-written
run_sft.sh: it reads the YAML, resolves the training mix to manifest paths, sets
the environment (frame budget, offline flags, GPU list, ``QALIGN_CFG`` for the
callback/shim), auto-enables the packed-image cache + in-training eval callback
when their artifacts exist, builds the ``swift sft`` argv, and runs it (tee'd to a
log). swift/torch are never imported here — we shell out to the ``swift`` CLI.

  mode="full"  full run per the config (epochs, save cadence, eval on)
  mode="mini"  10-step smoke test (grad_accum forced to 1, eval off by default)
"""
import os, sys, subprocess


def _faithful_lr(cfg):
    if cfg.train.lr is not None:
        return cfg.train.lr
    return 2e-4 if cfg.train.train_type == "lora" else 2e-5


def build_command(cfg, mode="full"):
    """Return (argv, env, out_dir, log_path) for the swift sft invocation."""
    t = cfg.train
    # 1) training mix -> manifest paths
    manifests = [cfg.manifest_path(n) for n in cfg.data.mix]
    if not manifests:
        raise ValueError("data.mix is empty — list the dataset names to train on")
    for m in manifests:
        if mode == "full" and (not os.path.exists(m) or os.path.getsize(m) == 0):
            raise FileNotFoundError(f"manifest missing/empty: {m} "
                                    "(run `qalign build` / `qalign frames` first)")

    # 2) output dir, tagged so variants don't clobber each other
    vit = "vitfrozen" if t.freeze_vit else "vittrained"
    tag = f"{'+'.join(cfg.data.mix)}_{t.train_type}_{vit}"
    out_dir = os.path.join(cfg.paths.output_dir, ("mini_" if mode == "mini" else "full_") + tag)
    os.makedirs(out_dir, exist_ok=True)

    lr = _faithful_lr(cfg)
    ga = 1 if mode == "mini" else t.grad_accum            # mini set is tiny vs max_steps

    # 3) method args
    if t.train_type == "lora":
        method = ["--tuner_type", "lora", "--lora_rank", str(t.lora_rank),
                  "--lora_alpha", str(t.lora_alpha), "--target_modules", "all-linear"]
    else:
        method = ["--tuner_type", "full", "--deepspeed", t.deepspeed]

    # 4) save / step args
    if mode == "mini":
        steps = ["--max_steps", "10", "--save_steps", "10", "--num_train_epochs", "1"]
    else:
        steps = ["--num_train_epochs", str(t.epochs),
                 "--save_steps", str(t.save_steps),
                 "--save_total_limit", str(t.save_total_limit)]
        if t.max_steps:
            steps += ["--max_steps", str(t.max_steps)]
    if t.save_only_model:
        steps += ["--save_only_model", "true"]
    if t.resume_from_checkpoint:
        steps += ["--resume_from_checkpoint", t.resume_from_checkpoint,
                  "--resume_only_model", "true"]

    # 5) register shim (cache hook + eval callback) — only when artifacts/eval ask for it
    register, callbacks = [], []
    shim = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_swift_register.py")
    want_cache = os.path.exists(os.path.join(cfg.paths.cache_dir, "index.json"))
    want_eval = cfg.eval.in_training and mode != "mini"
    if want_cache or want_eval:
        register = ["--custom_register_path", shim]
    if want_eval:
        callbacks = ["--callbacks", "qalign_eval"]

    dl = t.dataloader
    argv = [
        "swift", "sft",
        "--model", cfg.model.path,
        "--model_type", cfg.model.model_type,
        "--template", cfg.model.template,
        "--enable_thinking", str(cfg.model.enable_thinking).lower(),
        *register, *callbacks, *method,
        "--bf16", "true",
        "--dataset", *manifests,
        "--split_dataset_ratio", "0",
        "--per_device_train_batch_size", str(t.batch_size),
        "--gradient_accumulation_steps", str(ga),
        "--learning_rate", str(lr),
        "--freeze_vit", str(t.freeze_vit).lower(),
        "--freeze_aligner", str(t.freeze_aligner).lower(),
        "--max_length", str(cfg.model.max_length),
        "--warmup_ratio", str(t.warmup_ratio),
        "--lr_scheduler_type", t.lr_scheduler,
        "--logging_steps", str(t.logging_steps),
        "--dataset_num_proc", str(dl.dataset_num_proc),
        "--dataloader_num_workers", str(dl.num_workers),
        "--dataloader_prefetch_factor", str(dl.prefetch),
        "--dataloader_persistent_workers", "true",
        "--dataloader_pin_memory", "true",
        "--gradient_checkpointing", "true",
        "--attn_impl", cfg.model.attn_impl,
        "--output_dir", out_dir,
        *steps,
    ]

    env = dict(os.environ)
    env["QALIGN_CFG"] = getattr(cfg, "_yaml_path", env.get("QALIGN_CFG", ""))
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["NPROC_PER_NODE"] = str(t.gpus)
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(t.gpus)))
    env["MAX_NUM_FRAMES"] = str(cfg.data.frames_per_video)
    env["VIDEO_MAX_PIXELS"] = str(360 * 420)
    env["FPS"] = "1"
    return argv, env, out_dir, os.path.join(out_dir, "train.log")


def run(cfg, mode="full", dry_run=False):
    argv, env, out_dir, log = build_command(cfg, mode)
    print("[qalign train] output_dir:", out_dir, file=sys.stderr)
    print("[qalign train] command:\n  " + " ".join(argv), file=sys.stderr)
    if dry_run:
        return 0
    with open(log, "w") as lf:
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:                          # tee to console + log
            sys.stdout.write(line)
            lf.write(line)
        return proc.wait()
