import optuna

from src.tuning.config import SearchParameterConfig, IntSearchConfig, FloatSearchConfig, CategoricalSearchConfig


class SearchSpace:
    """Generate candidate hyperparameters."""

    def __init__(self, config: dict[str, SearchParameterConfig]) -> None:
        self.config = config

    def suggest(self, trial: optuna.Trial) -> dict:
        params = {}

        for name, space in self.config.items():
            if isinstance(space, IntSearchConfig):
                params[name] = trial.suggest_int(name, low=space.low, high=space.high, log=space.log)
            elif isinstance(space, FloatSearchConfig):
                params[name] = trial.suggest_float(name, low=space.low, high=space.high, log=space.log)
            elif isinstance(space, CategoricalSearchConfig):
                params[name] = trial.suggest_categorical(name, choices=space.choices)
            else:
                raise ValueError(f"Unsupported search space: {name}")
        return params