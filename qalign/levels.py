"""
Discrete quality levels — the heart of the Q-Align method.

Q-Align scores visual quality NOT as a regressed number but as one of K ordered
*text* levels (default 5: excellent > good > fair > poor > bad). Training teaches
the VL model to emit the level word; scoring reads the model's probability over
the K level tokens and takes a weighted average -> a continuous score.

This module is the single source of truth for:
  - the level vocabulary (BEST -> WORST) and its scalar weights, and
  - mapping a raw continuous MOS onto a level word (for datasets that ship MOS
    rather than pre-written Q-Align answers).

Everything here is plain Python (no torch / no model) and fully configurable from
YAML via `LevelScheme.from_cfg`.
"""
from dataclasses import dataclass, field
from typing import List

# Q-Align defaults, ordered BEST -> WORST, with the canonical weights. The weighted
# average of the level-token softmax with these weights is the continuous score.
DEFAULT_NAMES: List[str] = ["excellent", "good", "fair", "poor", "bad"]
DEFAULT_WEIGHTS: List[float] = [1.0, 0.75, 0.5, 0.25, 0.0]


@dataclass
class LevelScheme:
    """An ordered set of quality levels (BEST first) and their scalar weights."""
    names: List[str] = field(default_factory=lambda: list(DEFAULT_NAMES))
    weights: List[float] = field(default_factory=lambda: list(DEFAULT_WEIGHTS))

    def __post_init__(self):
        if len(self.names) != len(self.weights):
            raise ValueError(
                f"levels: names ({len(self.names)}) and weights ({len(self.weights)}) "
                "must have equal length")

    @classmethod
    def from_cfg(cls, cfg) -> "LevelScheme":
        """Build from a LevelsCfg (or any object exposing .names / .weights)."""
        return cls(names=list(cfg.names), weights=list(cfg.weights))

    # --- MOS -> level word --------------------------------------------------
    def map_score(self, score: float, lo: float, hi: float, dmos: bool = False) -> str:
        """Bin a raw score in [lo, hi] onto a level word (equal-width binning).

        Higher score -> better level by default. Set ``dmos=True`` for sets where
        higher = worse (Differential MOS, e.g. LIVE / CSIQ), which inverts the map.
        """
        if hi > lo:
            t = (score - lo) / (hi - lo)
        else:
            t = 0.0
        t = min(1.0, max(0.0, t))
        k = len(self.names)
        idx = min(k - 1, int(t * k))          # ascending-quality bin: 0 = worst region
        if dmos:                              # higher score == worse -> flip
            idx = k - 1 - idx
        # self.names is BEST->WORST; convert ascending-quality index to a name.
        return self.names[k - 1 - idx]


def default_scheme() -> LevelScheme:
    return LevelScheme()
