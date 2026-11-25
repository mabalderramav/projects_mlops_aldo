import logging

from src.common.logging import setup_logging
from src.settings import Settings
from src.data.pipeline import DataPipeline


logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging()

    try:
        settings = Settings()

        logging.getLogger().setLevel(settings.log_level)
        logger.info("Application environment: %s", settings.app_env)

        pipeline = DataPipeline(
            config_path=settings.data_config_path
        )
        pipeline.run()

    except Exception:
        logger.exception("Data pipeline execution failed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()