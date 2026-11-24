import json
import logging

from src.common.logging import setup_logging
from src.settings import Settings
from src.evaluation.pipeline import EvaluationPipeline


logger = logging.getLogger(__name__)

def main() -> None:
    setup_logging()

    try:
        settings = Settings()

        logging.getLogger().setLevel(settings.log_level)
        logger.info("Evaluation environment: %s", settings.app_env)

        pipeline = EvaluationPipeline(
            config_path=settings.evaluation_config_path
        )
        metrics = pipeline.run()

        print("\nEvaluation Metrics\n")
        print(json.dumps(metrics, indent=4))

    except Exception:
        logger.exception("Evaluation execution failed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()