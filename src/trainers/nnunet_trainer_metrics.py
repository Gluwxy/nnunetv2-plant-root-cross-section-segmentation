"""
nnU-Net v2 trainers that log more segmentation metrics than the default nnUNetTrainer.

nnU-Net only logs the pseudo Dice per class during training and the mean Dice of the final validation.
These trainers also log, per class (by name) and as the mean over the foreground classes:
- every epoch, from the TP/FP/FN that nnU-Net counts on the validation patches: Dice, IoU, precision
  and recall (W&B keys val_<metric>/<class>; the pseudo IoU is also written to the training log)
- after the final validation on the full images (validation/summary.json): the same metrics averaged
  over cases (W&B summary keys final_val/<metric>/<class>) and a per-case table (final_val/per_case)

nnU-Net finds these classes through the nnUNet_extTrainer environment variable (set by src/main.py, the
Dockerfile and .envrc) and imports this file as a top-level module, so it must not import from `src`.
Training length follows nnU-Net's naming: nnUNetTrainerMetrics (1000 epochs), nnUNetTrainerMetrics_20epochs...
"""
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json
from nnunetv2.evaluation.evaluate_predictions import label_or_region_to_key
from nnunetv2.training.logging.nnunet_logger import WandbLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs

# Same lengths as nnU-Net's nnUNetTrainer_<N>epochs variants
EPOCH_VARIANTS = (1, 2, 5, 10, 20, 50, 100, 250, 500, 750, 2000, 4000, 8000)
# Default device of nnUNetTrainer.__init__
CUDA = torch.device("cuda")


def segmentation_metrics(tp, fp, fn) -> dict[str, np.ndarray]:
    """Per-class Dice, IoU, precision and recall from TP/FP/FN counts (NaN where a class is absent)."""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "dice": 2 * tp / (2 * tp + fp + fn),
            "iou": tp / (tp + fp + fn),
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
        }


class nnUNetTrainerMetrics(nnUNetTrainer):
    def _labels_or_regions(self) -> list:
        lm = self.label_manager
        return list(lm.foreground_regions if lm.has_regions else lm.foreground_labels)

    def _class_names(self) -> list[str]:
        """Class names from dataset.json (e.g. root, stele, xylem), in the order nnU-Net reports them."""
        if self.label_manager.has_regions:
            return [f"region_{i + 1}" for i in range(len(self._labels_or_regions()))]
        value_to_name = {v: k for k, v in self.dataset_json["labels"].items()}
        return [value_to_name.get(v, f"class_{v}") for v in self._labels_or_regions()]

    def _wandb_loggers(self) -> list[WandbLogger]:
        # nnU-Net's local logger only accepts its own keys, so extra metrics go to the W&B loggers directly
        return [lg for lg in self.logger.loggers if isinstance(lg, WandbLogger)]

    @staticmethod
    def _sum_over_ranks(x: np.ndarray) -> np.ndarray:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, x)
        return np.sum(gathered, 0)

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)

        outputs = collate_outputs(val_outputs)
        counts = [np.sum(outputs[k], 0) for k in ("tp_hard", "fp_hard", "fn_hard")]
        if self.is_ddp:
            counts = [self._sum_over_ranks(c) for c in counts]
        metrics = segmentation_metrics(*counts)

        self.print_to_log_file("Pseudo IoU", [np.round(i, decimals=4) for i in metrics["iou"]])
        names = self._class_names()
        for wandb_logger in self._wandb_loggers():
            for metric, values in metrics.items():
                for name, value in zip(names, values, strict=True):
                    wandb_logger.log(f"val_{metric}/{name}", float(value), self.current_epoch)
                wandb_logger.log(f"val_{metric}/mean", float(np.nanmean(values)), self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)

        summary_file = join(self.output_folder, "validation", "summary.json")
        wandb_loggers = self._wandb_loggers()
        if self.global_rank != 0 or not wandb_loggers or not isfile(summary_file):
            return

        import wandb

        names = self._class_names()
        keys = [label_or_region_to_key(lr) for lr in self._labels_or_regions()]
        cases = load_json(summary_file)["metric_per_case"]

        # Metrics per case and class, from the full-image TP/FP/FN of nnU-Net's final validation
        per_case = {}  # case -> metric -> per-class values
        for case in cases:
            counts = [[case["metrics"][k][c] for k in keys] for c in ("TP", "FP", "FN")]
            per_case[Path(case["reference_file"]).stem] = segmentation_metrics(*counts)

        metric_names = ("dice", "iou", "precision", "recall")
        columns = ["case"] + [f"{metric}/{name}" for metric in metric_names for name in names]
        table = wandb.Table(columns=columns)
        for case, metrics in per_case.items():
            table.add_data(case, *[float(v) for metric in metric_names for v in metrics[metric]])

        for wandb_logger in wandb_loggers:
            for metric in metric_names:
                class_means = np.nanmean([m[metric] for m in per_case.values()], axis=0)
                for name, value in zip(names, class_means, strict=True):
                    wandb_logger.log_summary(f"final_val/{metric}/{name}", float(value))
                wandb_logger.log_summary(f"final_val/{metric}/mean", float(np.nanmean(class_means)))
            wandb_logger.run.log({"final_val/per_case": table})


def _epoch_variant(num_epochs: int) -> type:
    """nnUNetTrainerMetrics subclass that trains for num_epochs (like nnU-Net's nnUNetTrainer_<N>epochs)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = CUDA):
        # Same argument names as nnUNetTrainer: it stores them (inspect.signature) to rebuild the trainer
        nnUNetTrainerMetrics.__init__(self, plans, configuration, fold, dataset_json, device)
        self.num_epochs = num_epochs

    suffix = "1epoch" if num_epochs == 1 else f"{num_epochs}epochs"
    return type(f"nnUNetTrainerMetrics_{suffix}", (nnUNetTrainerMetrics,),
                {"__init__": __init__, "__module__": __name__})


# Module-level names, so nnU-Net can find them with getattr(module, trainer_name)
for _n in EPOCH_VARIANTS:
    _cls = _epoch_variant(_n)
    globals()[_cls.__name__] = _cls
