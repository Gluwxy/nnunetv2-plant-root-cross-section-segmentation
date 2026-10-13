import os
import rootutils
from pathlib import Path
from roboflow import Roboflow

import hydra
from omegaconf import DictConfig

rootutils.setup_root(Path.cwd(), indicator=".git", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_rf = cfg.roboflow

    rf_api_key = os.environ.get("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=rf_api_key)
    project = rf.workspace(cfg_rf.workspace).project(cfg_rf.project)
    version = project.version(cfg_rf.version)
    dataset = version.download(cfg_rf.format, location=cfg_rf.location)


if __name__ == "__main__":
    main()
