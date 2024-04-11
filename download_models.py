import os
from pathlib import Path

import invokeai.app.services.config

from invokeai.app.invocations.upscale import ESRGAN_MODEL_URLS
from invokeai.app.services.download import DownloadQueueService
from invokeai.app.util.download_with_progress import download_with_progress_bar
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.model_install import ModelInstallService
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger

models = [
    {
        "repo": "dreamlike-art/dreamlike-photoreal-2.0",
        "dir": "sd-1/main/dreamlike-photoreal-2.0",
    },
    {
        "repo": "Lykon/dreamshaper-8-inpainting",
        "dir": "sd-1/main/dreamshaper-8-inpainting",
    },
    {
        "repo": "runwayml/stable-diffusion-inpainting",
        "dir": "sd-1/main/stable-diffusion-inpainting",
    },
    {
        "repo": "runwayml/stable-diffusion-v1-5",
        "dir": "sd-1/main/stable-diffusion-v1-5",
    },
    {
        "repo": "stabilityai/stable-diffusion-2-1",
        "dir": "sd-2/main/stable-diffusion-2-1",
    },
]


def get_installer(_config, _logger) -> ModelInstallService:
    image_files = DiskImageFileStorage(f"{_config.root_path}/outputs/images")
    db = init_db(config=_config, logger=_logger, image_files=image_files)
    record_store = ModelRecordServiceSQL(db)
    queue = DownloadQueueService()
    queue.start()

    return ModelInstallService(app_config=_config,
                               record_store=record_store,
                               download_queue=queue
                              )


if __name__ == "__main__":
    config = invokeai.app.services.config.get_config()
    logger = InvokeAILogger.get_logger(config=config)

    installer = get_installer(config, logger)
    installer.start()

    models_dir = Path(config.root_path, config.models_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    for model in models:
        local_dir = Path(models_dir, model["dir"])
        if not os.path.exists(local_dir):
            installer.heuristic_import(model["repo"])

    for upscaler in ["RealESRGAN_x4plus.pth"]:
        local_dir = Path(models_dir, f"core/upscaling/realesrgan/{upscaler}")
        download_with_progress_bar(upscaler, ESRGAN_MODEL_URLS[upscaler], local_dir)

    installer.wait_for_installs()
    installer.stop()
