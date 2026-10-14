import logging
import os

import hydra
import rootutils
from omegaconf import DictConfig
from roboflow import Roboflow

from src.utils.resolvers import get_splits  # also registers the OmegaConf resolvers used in cfg/

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_rf = cfg.roboflow

    # Roboflow skips the download whenever the folder exists, but the Docker targets create it as a
    # mount point: download unless it already holds a split with COCO annotations
    existing = get_splits(cfg_rf.location)
    if existing:
        log.info(f"Dataset already in {cfg_rf.location} (splits {list(existing)}), skipping download")
        return

    rf_api_key = os.environ.get("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=rf_api_key)
    project = rf.workspace(cfg_rf.workspace).project(cfg_rf.project)
    version = project.version(cfg_rf.version)
    version.download(cfg_rf.format, location=cfg_rf.location, overwrite=True)


if __name__ == "__main__":
    main()
