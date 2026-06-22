"""
Pure-Python unit tests (no GPU, no swift) for the model-agnostic core:
levels, prompts, template, config loading, and CSV manifest building.

Run:  pytest -q
"""
import os, json
import pytest

from qalign import Config, LevelScheme, TaskPrompts, make_record
from qalign import datasets as D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXAMPLE_CFG = os.path.join(REPO, "configs", "example_iqa.yaml")


# --- levels ----------------------------------------------------------------
def test_level_mapping_equal_width():
    s = LevelScheme()                       # excellent..bad, weights 1..0
    # MOS in [1, 5]: top of range -> best, bottom -> worst, middle -> fair
    assert s.map_score(5.0, 1.0, 5.0) == "excellent"
    assert s.map_score(1.0, 1.0, 5.0) == "bad"
    assert s.map_score(3.0, 1.0, 5.0) == "fair"


def test_level_mapping_monotonic():
    s = LevelScheme()
    order = {n: i for i, n in enumerate(s.names)}      # 0 = best
    prev = None
    for mos in [1.0, 2.0, 3.0, 4.0, 5.0]:
        rank = order[s.map_score(mos, 1.0, 5.0)]
        if prev is not None:
            assert rank <= prev                         # higher MOS -> better-or-equal level
        prev = rank


def test_level_mapping_dmos_inverts():
    s = LevelScheme()
    # DMOS: higher == worse, so the top of the range maps to the worst level
    assert s.map_score(5.0, 1.0, 5.0, dmos=True) == "bad"
    assert s.map_score(1.0, 1.0, 5.0, dmos=True) == "excellent"


def test_levels_length_mismatch_raises():
    with pytest.raises(ValueError):
        LevelScheme(names=["a", "b"], weights=[1.0])


# --- prompts ---------------------------------------------------------------
def test_task_prompts_answer():
    tp = TaskPrompts.for_task("iqa")
    assert tp.answer("good") == "The quality of the image is good."
    assert TaskPrompts.for_task("vqa").stem.endswith("video is")
    assert len(tp.prompts) > 0


def test_task_prompts_override():
    tp = TaskPrompts.for_task("iqa", {"iqa": {"stem": "Quality:", "prompts": ["rate it"]}})
    assert tp.answer("bad") == "Quality: bad."
    assert tp.prompts == ["rate it"]


# --- template --------------------------------------------------------------
def test_make_record_image():
    r = make_record(["/x/a.jpg"], "Rate it.", "The quality of the image is good.")
    assert r["messages"][0]["content"].count("<image>") == 1
    assert r["messages"][0]["role"] == "user"
    assert r["messages"][1]["role"] == "assistant"
    assert r["images"] == ["/x/a.jpg"]


def test_make_record_video_eight_frames():
    frames = [f"/x/f{i}.jpg" for i in range(8)]
    r = make_record(frames, "Rate it.", "The quality of the video is fair.")
    assert r["messages"][0]["content"].count("<image>") == 8
    assert len(r["images"]) == 8


def test_make_record_empty_raises():
    with pytest.raises(ValueError):
        make_record([], "p", "a")


# --- config ----------------------------------------------------------------
def test_example_config_loads():
    cfg = Config.from_yaml(EXAMPLE_CFG)
    assert cfg.model.model_type == "qwen3_5"
    assert cfg.levels.names[0] == "excellent"
    assert cfg.data.mix == ["toy_iqa"]
    assert cfg.dataset("toy_iqa").task == "iqa"
    # ${root} expansion + abspath
    assert os.path.isabs(cfg.paths.data_dir)
    assert cfg.paths.data_dir.endswith(os.path.join("examples", "runs", "data"))


def test_config_override():
    cfg = Config.from_yaml(EXAMPLE_CFG, overrides=["train.lr=1e-5", "train.gpus=4"])
    assert cfg.train.lr == 1e-5
    assert cfg.train.gpus == 4


# --- datasets (CSV -> manifests) -------------------------------------------
def test_build_train_and_eval_manifest(tmp_path):
    cfg = Config.from_yaml(EXAMPLE_CFG, overrides=[f"paths.root={tmp_path.as_posix()}"])
    # the example CSV/imgs live under the repo; run from repo root so relative paths resolve
    os.chdir(REPO)
    tr = D.build_train_manifest(cfg, "toy_iqa")
    assert tr["kept"] == 12 and tr["missing"] == 0
    recs = [json.loads(l) for l in open(tr["out"])]
    # every record is a leveled IQA conversation with one image
    assert all(r["messages"][0]["content"].count("<image>") == 1 for r in recs)
    assert all(r["messages"][1]["content"].startswith("The quality of the image is") for r in recs)

    ev = D.build_eval_manifest(cfg, "toy_iqa")
    assert ev["kept"] == 12
    erecs = [json.loads(l) for l in open(ev["out"])]
    assert all(set(e) >= {"modality", "src", "prompt", "stem", "gt_score"} for e in erecs)
