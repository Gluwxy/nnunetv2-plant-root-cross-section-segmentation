import os
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig
from roboflow import Roboflow

rootutils.setup_root(Path.cwd(), indicator="pyproject.toml", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_rf = cfg.roboflow

    rf_api_key = os.environ.get("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=rf_api_key)
    project = rf.workspace(cfg_rf.workspace).project(cfg_rf.project)
    version = project.version(cfg_rf.version)
    version.download(cfg_rf.format, location=cfg_rf.location)


if __name__ == "__main__":
    main()
