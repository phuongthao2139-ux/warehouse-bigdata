"""Locked chronological protocol (docs/07 section 8) and shared constants."""

from dataclasses import dataclass

HORIZON = 7
LOOKBACK_DAYS = 28
FEATURE_VERSION = "rf_lag_v1"

RUN_TYPES = ("validation", "test", "forward_demo")
METHOD_RANDOM_FOREST = "random_forest"
METHOD_SEASONAL_NAIVE = "seasonal_naive"


@dataclass(frozen=True, slots=True)
class Protocol:
    """Day-index boundaries for training, validation, and locked test periods.

    Origins are issue days ``T``; each scores the next seven days.
    """

    train_last_day: int = 1885
    validation_last_day: int = 1913
    test_last_day: int = 1941
    validation_origins: tuple[int, ...] = (1885, 1892, 1899, 1906)
    test_origins: tuple[int, ...] = (1913, 1920, 1927, 1934)

    def role_for_day(self, day_index: int) -> str:
        """Return the feature role for a target day."""

        if day_index <= self.train_last_day:
            return "train"
        if day_index <= self.validation_last_day:
            return "validation"
        return "test"

    def origins_for(self, run_type: str) -> tuple[int, ...]:
        if run_type == "validation":
            return self.validation_origins
        if run_type == "test":
            return self.test_origins
        raise ValueError(f"run_type must be 'validation' or 'test' for scored origins: {run_type}")


M5_PROTOCOL = Protocol()
