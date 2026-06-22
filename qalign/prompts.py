"""
Task prompt + answer-stem library.

Q-Align routes three quality-assessment tasks through ONE model by varying the
answer stem the assistant completes with a level word:

  iqa  image quality      "The quality of the image is <level>."
  iaa  image aesthetics   "The aesthetics of the image is <level>."
  vqa  video quality       "The quality of the video is <level>."

Each task also carries a pool of paraphrased human prompts (one is picked per
sample) so the model isn't overfit to a single phrasing. The five level words are
shared across tasks by design (see qalign/levels.py).

The IQA prompts are taken verbatim from Q-Align; the IAA / VQA pools are
Q-Align-style paraphrases. ALL of this is overridable from YAML — see
``TaskPrompts.from_cfg`` and the ``prompts:`` section of a config — so adapting to
a new task / language / phrasing is a config edit, not a code change.
"""
from dataclasses import dataclass
from typing import Dict, List

# stem the assistant completes with the level word (the model "routes" on this)
STEMS: Dict[str, str] = {
    "iqa": "The quality of the image is",
    "iaa": "The aesthetics of the image is",
    "vqa": "The quality of the video is",
}

PROMPTS: Dict[str, List[str]] = {
    "iqa": [
        "Can you rate the quality of this picture?",
        "Could you evaluate the quality of this image?",
        "How do you assess the quality of this image?",
        "How would you judge the quality of this image?",
        "How would you rate the quality of this image?",
        "Rate the quality of this image.",
        "What do you think about the quality of this image?",
        "What is your quality rating for this image?",
        "What's your opinion on the quality of this picture?",
    ],
    "iaa": [
        "How is the aesthetics of this image?",
        "Could you evaluate the aesthetic quality of this image?",
        "How would you rate the aesthetics of this image?",
        "How would you judge the aesthetic appeal of this picture?",
        "Rate the aesthetics of this image.",
        "What do you think about the aesthetics of this image?",
        "What is your aesthetic rating for this image?",
        "How aesthetically pleasing is this image?",
    ],
    "vqa": [
        "Can you rate the quality of this video?",
        "Could you evaluate the quality of this video?",
        "How do you assess the quality of this video?",
        "How would you judge the quality of this video?",
        "How would you rate the quality of this video?",
        "Rate the quality of this video.",
        "What do you think about the quality of this video?",
        "What is your quality rating for this video?",
    ],
}


@dataclass
class TaskPrompts:
    """Prompt pool + answer stem for one task."""
    task: str
    stem: str
    prompts: List[str]

    def answer(self, level_word: str) -> str:
        """The full assistant target: '<stem> <level>.'"""
        return f"{self.stem} {level_word}."

    @classmethod
    def for_task(cls, task: str, overrides: dict = None) -> "TaskPrompts":
        if task not in STEMS:
            raise ValueError(f"unknown task '{task}' (known: {list(STEMS)})")
        stem = STEMS[task]
        prompts = list(PROMPTS[task])
        if overrides:
            o = overrides.get(task, {})
            stem = o.get("stem", stem)
            prompts = list(o.get("prompts", prompts))
        return cls(task=task, stem=stem, prompts=prompts)
