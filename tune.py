import json
import logging

from src.common.logging import setup_logging
from src.settings import Settings
from src.tuning.pipeline import TuningPipeline


logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging()

    try:
        settings = Settings()

        logging.getLogger().setLevel(settings.log_level)
        logger.info("Application environment: %s", settings.app_env)

        pipeline = TuningPipeline(config_path=settings.tuning_config_path)
        results = pipeline.run()

        print("\nTuning Results\n")
        print(json.dumps(results, indent=4))

    except Exception:
        logger.exception("Tuning pipeline execution failed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()