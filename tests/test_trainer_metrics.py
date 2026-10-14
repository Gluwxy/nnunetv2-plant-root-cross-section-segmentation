"""Unit tests for the custom nnU-Net trainers in src/trainers/."""
import inspect
from pathlib import Path

import numpy as np
import pytest
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

from src.trainers.nnunet_trainer_metrics import EPOCH_VARIANTS, segmentation_metrics

TRAINERS_DIR = Path(__file__).resolve().parents[1] / "src" / "trainers"


def test_segmentation_metrics():
    # class 1: tp=6, fp=2, fn=4 | class 2: absent from prediction and reference
    m = segmentation_metrics([6, 0], [2, 0], [4, 0])
    assert m["dice"][0] == pytest.approx(12 / 18)
    assert m["iou"][0] == pytest.approx(6 / 12)
    assert m["precision"][0] == pytest.approx(6 / 8)
    assert m["recall"][0] == pytest.approx(6 / 10)
    assert all(np.isnan(m[k][1]) for k in m)
    # IoU and Dice of the same counts are linked: IoU = D / (2 - D)
    assert m["iou"][0] == pytest.approx(m["dice"][0] / (2 - m["dice"][0]))


@pytest.mark.parametrize("name, epochs", [("nnUNetTrainerMetrics", 1000),
                                          ("nnUNetTrainerMetrics_1epoch", 1),
                                          ("nnUNetTrainerMetrics_20epochs", 20),
                                          ("nnUNetTrainerMetrics_250epochs", 250)])
def test_nnunet_finds_metric_trainers(monkeypatch, name, epochs):
    """nnUNetv2_train/predict must find the trainers through nnUNet_extTrainer (as set by src/main.py)."""
    monkeypatch.setenv("nnUNet_extTrainer", str(TRAINERS_DIR))
    trainer = recursive_find_trainer_class_by_name(name)
    assert issubclass(trainer, nnUNetTrainer)
    assert trainer.__name__ == name
    # nnUNetTrainer stores its __init__ arguments by name to rebuild the trainer, so they must match
    assert list(inspect.signature(trainer.__init__).parameters) == \
        list(inspect.signature(nnUNetTrainer.__init__).parameters)
    if epochs != 1000:
        assert epochs in EPOCH_VARIANTS
