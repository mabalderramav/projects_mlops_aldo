import json
import logging

from src.common.logging import setup_logging
from src.settings import Settings
from src.validation.pipeline import ValidationPipeline


logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging()

    try:
        settings = Settings()

        logging.getLogger().setLevel(settings.log_level)
        logger.info("Application environment: %s", settings.app_env)

        pipeline = ValidationPipeline(config_path=settings.validation_config_path)
        metrics = pipeline.run()

        print("\nValidation Metrics\n")
        print(json.dumps(metrics, indent=4))

    except Exception:
        logger.exception("Validation pipeline execution failed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()