"""
qalign — modernized, model-agnostic Q-Align.

Visual quality / aesthetics scoring as discrete text levels
(excellent / good / fair / poor / bad), trainable and evaluable on ANY
ms-swift-supported vision-language model via a single YAML config.

Light, laptop-importable surface (no torch / no swift at import time):
    from qalign import Config, LevelScheme, TaskPrompts, make_record

The training / eval / inference modules (qalign.model, qalign.scorer,
qalign.train, qalign.infer, qalign.callback) import ms-swift / torch lazily, so
they are only required on the GPU box.
"""
from .config import Config
from .levels import LevelScheme, default_scheme, DEFAULT_NAMES, DEFAULT_WEIGHTS
from .prompts import TaskPrompts, STEMS, PROMPTS
from .template import make_record, write_manifest

__version__ = "0.1.0"

__all__ = [
    "Config",
    "LevelScheme", "default_scheme", "DEFAULT_NAMES", "DEFAULT_WEIGHTS",
    "TaskPrompts", "STEMS", "PROMPTS",
    "make_record", "write_manifest",
    "__version__",
]
