"""
Experiment configuration: one YAML fully describes a run.

A single ``Config`` (nested dataclasses) captures the model, the level scheme,
prompt overrides, filesystem layout, the dataset sources + training mix, the
training hyper-parameters, and the eval settings. Adapting Q-Align to a new model
or a new dataset is editing this YAML — no code change.

Load with ``Config.from_yaml(path, overrides=["train.lr=1e-5", ...])``. Defaults
are faithful to Q-Align (5 levels, 8 frames, cosine LR, warmup 0.03, 2 epochs,
full-parameter FT with the vision tower + projector trainable).

This module is plain Python (yaml + dataclasses); importing it pulls in no torch.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Dict, Any, Optional
import os

from .levels import DEFAULT_NAMES, DEFAULT_WEIGHTS


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class ModelCfg:
    path: str = ""                 # local path or HF id of the base VL model
    model_type: str = "qwen3_5"    # swift model_type (any registered VL model)
    template: str = "qwen3_5"      # swift template name (usually == model_type)
    enable_thinking: bool = False  # qwen "thinking" toggle; ignored by templates without it
    max_length: int = 8192         # 8 frames/video reach ~7k tokens — do not lower for video
    attn_impl: str = "flash_attn"  # flash_attn | sdpa | eager


@dataclass
class LevelsCfg:
    names: List[str] = field(default_factory=lambda: list(DEFAULT_NAMES))
    weights: List[float] = field(default_factory=lambda: list(DEFAULT_WEIGHTS))


@dataclass
class PathsCfg:
    root: str = "."                # working dir; the other paths default under it
    data_dir: str = "${root}/data"            # built training manifests
    manifest_dir: str = "${root}/eval_manifests"   # built eval manifests
    frames_dir: str = "${root}/frames"        # extracted video frames
    cache_dir: str = "${root}/cache"          # packed TRAINING image blob
    eval_cache_dir: str = "${root}/eval_cache"# packed EVAL image blob
    output_dir: str = "${root}/output"        # checkpoints + logs


@dataclass
class DatasetEntry:
    name: str                      # referenced by data.mix / eval.sets
    task: str = "iqa"              # iqa | iaa | vqa  -> picks stem + prompt pool
    format: str = "csv"            # csv | jsonl | qalign_json
    source: str = ""               # the labels file (csv / jsonl / Q-Align json)
    media_root: str = ""           # prefix joined to each relative media path
    modality: str = "image"        # image | video
    # --- column / key names (csv & jsonl) ---
    path_col: str = "path"         # media path column/key
    score_col: str = "mos"         # raw MOS column/key (when no pre-written answer)
    answer_col: str = ""           # optional: a pre-written Q-Align answer column/key
    # --- score handling ---
    dmos: bool = False             # true if higher score == worse (e.g. LIVE / CSIQ)
    score_min: Optional[float] = None  # binning range; if None, computed from the data
    score_max: Optional[float] = None
    basename_only: bool = False    # resolve media by basename under media_root (e.g. KADID)


@dataclass
class DataCfg:
    frames_per_video: int = 8
    resize_long: int = 448
    datasets: List[DatasetEntry] = field(default_factory=list)
    mix: List[str] = field(default_factory=list)   # dataset names to TRAIN on


@dataclass
class DataloaderCfg:
    num_workers: int = 24
    prefetch: int = 6
    dataset_num_proc: int = 32


@dataclass
class TrainCfg:
    train_type: str = "full"       # full (param FT) | lora
    deepspeed: str = "zero2"       # zero2 | zero3 | zero3_offload (full only)
    lr: Optional[float] = None     # default 2e-5 (full) / 2e-4 (lora) if None
    epochs: int = 2
    batch_size: int = 4            # per-device
    grad_accum: int = 2
    gpus: int = 2
    freeze_vit: bool = False       # train the vision tower (onealign-faithful)
    freeze_aligner: bool = False   # train the projector/merger
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    save_steps: int = 500
    save_total_limit: int = 3
    save_only_model: bool = True   # drop optimizer state (weights-only checkpoints)
    logging_steps: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    max_steps: Optional[int] = None
    resume_from_checkpoint: str = ""
    dataloader: DataloaderCfg = field(default_factory=DataloaderCfg)


@dataclass
class EvalCfg:
    sets: List[str] = field(default_factory=list)  # dataset names to EVAL on
    limit: int = 200               # records/set during in-training eval (0 = all)
    in_training: bool = True       # register the in-training eval callback
    every: int = 1                 # eval every Nth checkpoint save
    keep_best_n: int = 1           # preserve top-N checkpoints (by avg SRCC) in best/
    use_ram_cache: bool = True     # pin the packed eval blob in RAM if present


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    levels: LevelsCfg = field(default_factory=LevelsCfg)
    prompts: Dict[str, Any] = field(default_factory=dict)   # per-task stem/prompts overrides
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    _yaml_path: str = field(default="", repr=False)

    # --- lookups ---
    def dataset(self, name: str) -> DatasetEntry:
        for d in self.data.datasets:
            if d.name == name:
                return d
        raise KeyError(f"dataset '{name}' not defined under data.datasets")

    def manifest_path(self, name: str) -> str:
        return os.path.join(self.paths.data_dir, f"{name}.jsonl")

    def eval_manifest_path(self, name: str) -> str:
        return os.path.join(self.paths.manifest_dir, f"{name}.jsonl")

    # --- construction ---
    @classmethod
    def from_yaml(cls, path: str, overrides: List[str] = None) -> "Config":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for ov in (overrides or []):
            _apply_override(raw, ov)
        cfg = _build(cls, raw)
        assert isinstance(cfg, Config)        # _build returns this class; help the type checker
        cfg._resolve_paths()
        cfg._yaml_path = os.path.abspath(path)
        return cfg

    def _resolve_paths(self):
        root = os.path.abspath(os.path.expanduser(self.paths.root))
        self.paths.root = root
        for fld in fields(self.paths):
            if fld.name == "root":
                continue
            val = getattr(self.paths, fld.name)
            val = val.replace("${root}", root)
            setattr(self.paths, fld.name, os.path.abspath(os.path.expanduser(val)))


# --------------------------------------------------------------------------- #
# Helpers: dict -> nested dataclass, dotted-key overrides
# --------------------------------------------------------------------------- #
def _coerce(value, hint):
    """Best-effort coerce a yaml scalar/list to a dataclass field type hint."""
    origin = getattr(hint, "__origin__", None)
    if origin in (list, List):
        (elem,) = getattr(hint, "__args__", (Any,))
        return [_coerce(v, elem) for v in (value or [])]
    return value


def _build(klass, data: dict):
    if not is_dataclass(klass):
        return data
    kwargs = {}
    type_hints = {f.name: f.type for f in fields(klass)}
    for f in fields(klass):
        if f.name not in (data or {}):
            continue
        raw = data[f.name]
        hint = type_hints[f.name]
        # nested dataclass (e.g. model:, train:, train.dataloader:)
        if is_dataclass(hint) and isinstance(raw, dict):
            kwargs[f.name] = _build(hint, raw)
        # list of dataclass (data.datasets -> List[DatasetEntry])
        elif getattr(hint, "__origin__", None) in (list, List):
            (elem,) = getattr(hint, "__args__", (Any,))
            if is_dataclass(elem) and isinstance(raw, list):
                kwargs[f.name] = [_build(elem, d) for d in raw]
            else:
                kwargs[f.name] = _coerce(raw, hint)
        else:
            kwargs[f.name] = raw
    return klass(**kwargs)


def _apply_override(raw: dict, override: str):
    """Apply a single ``a.b.c=value`` override into the raw dict (YAML-parsed value)."""
    import yaml
    if "=" not in override:
        raise ValueError(f"bad override '{override}' (expected key.path=value)")
    key, _, val = override.partition("=")
    node = raw
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    parsed = yaml.safe_load(val)              # parses true / [a,b] / 1.0e-5 correctly
    if isinstance(parsed, str):
        # YAML 1.1 doesn't treat '1e-5' (no dot) as a float — coerce numbers ourselves.
        try:
            parsed = int(parsed)
        except ValueError:
            try:
                parsed = float(parsed)
            except ValueError:
                pass
    node[parts[-1]] = parsed
