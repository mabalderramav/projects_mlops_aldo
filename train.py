import logging

from src.common.logging import setup_logging
from src.settings import Settings
from src.training.pipeline import TrainingPipeline


logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging()

    try:
        settings = Settings()

        logging.getLogger().setLevel(settings.log_level)
        logger.info("Application environment: %s", settings.app_env)

        pipeline = TrainingPipeline(
            config_path=settings.training_config_path
        )
        pipeline.run()

    except Exception:
        logger.exception("Application execution failed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()