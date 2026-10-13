"""
Plans, preprocesses and trains nnU-Net v2 through MONAI's nnUNetV2Runner (settings in cfg/monai/default.yaml).

The dataset must already exist in data/nnUNet_raw (run `uv run python -m src.utils.convert` first).

Usage:
    uv run python -m src.main                                 # preprocess + train fold 0, 20 epochs
    uv run python -m src.main monai.preprocess.enabled=false  # train only (already preprocessed)
    uv run python -m src.main monai.train.fold=1 monai.preprocess.enabled=false
    uv run python -m src.main monai.runner.trainer_class_name=nnUNetTrainer_250epochs
    uv run python -m src.main monai.train.device=mps          # Apple Silicon
"""
import logging
import os

import hydra
import rootutils
from hydra.utils import instantiate
from monai.utils import run_cmd
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    mcfg = cfg.monai
    os.environ["nnUNet_n_proc_DA"] = str(mcfg.train.n_proc_da)

    # Sets nnUNet_raw / nnUNet_preprocessed / nnUNet_results for this process and its subprocesses
    runner = instantiate(mcfg.runner)

    if mcfg.preprocess.enabled:
        pcfg = mcfg.preprocess
        # plan_and_process = extract_fingerprints + plan_experiments + preprocess
        runner.plan_and_process(
            npfp=pcfg.npfp,
            verify_dataset_integrity=pcfg.verify_dataset_integrity,
            c=tuple(pcfg.configs),
            n_proc=tuple(pcfg.n_proc),
        )

    if mcfg.train.enabled:
        tcfg = mcfg.train
        extra = {"c": True} if tcfg.continue_training else {}
        # Build the nnUNetv2_train command with the runner and add -device ourselves:
        # the runner prefixes extra args with "--", but nnUNetv2_train only accepts "-device".
        cmd, env = runner.train_single_model_command(tcfg.config, tcfg.fold, tcfg.gpu_id, extra)
        cmd += ["-device", str(tcfg.device)]
        log.info("Running: %s", " ".join(cmd))
        log.info("Config:\n%s", OmegaConf.to_yaml(mcfg))
        run_cmd(cmd, env=env, check=True, capture_output=False)  # stream logs, fail on error


if __name__ == "__main__":
    main()
