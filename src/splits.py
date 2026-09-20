"""Chronological train / validation / test / stress splits.

Split boundaries decide every reported number, so this module fails loudly
rather than adjusting. The configured dates imply exact lengths -- 5,472
training points, 1,008 each for validation and test, 1,440 for the stress
period -- and a mismatch means the series is not the one the configuration
describes. Silently returning a shorter split would change every metric without
raising anything.

Splits are contiguous and ordered in time. There is no shuffling and no random
partition: a forecasting model evaluated on data that precedes its training set
is measuring nothing.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.config import Config, DateRange

__all__ = ["Split", "SplitSet", "make_splits", "SplitError"]


class SplitError(ValueError):
    """Raised when the data does not match the configured split boundaries."""


@dataclass(frozen=True)
class Split:
    """One contiguous slice of a series, with its position in the whole."""

    name: str
    values: np.ndarray
    times: np.ndarray
    start: int
    stop: int

    def __len__(self) -> int:
        return int(self.values.size)

    @property
    def first_day(self) -> dt.date:
        return self.times[0].astype("datetime64[D]").astype(dt.date)

    @property
    def last_day(self) -> dt.date:
        return self.times[-1].astype("datetime64[D]").astype(dt.date)

    def describe(self) -> str:
        return (
            f"{self.name:<11} {len(self):>5} points  "
            f"{self.first_day} .. {self.last_day}  "
            f"[{self.start}:{self.stop}]"
        )


@dataclass(frozen=True)
class SplitSet:
    """The four splits of one area's series."""

    train: Split
    validation: Split
    test: Split
    stress: Split
    series: np.ndarray
    times: np.ndarray

    def __getitem__(self, name: str) -> Split:
        try:
            split = getattr(self, name)
        except AttributeError as exc:
            raise KeyError(f"unknown split {name!r}") from exc
        if not isinstance(split, Split):
            raise KeyError(f"unknown split {name!r}")
        return split

    def __iter__(self) -> Iterator[Split]:
        return iter((self.train, self.validation, self.test, self.stress))

    @property
    def fit_values(self) -> np.ndarray:
        """Train plus validation, used to refit before the final test run.

        Selecting hyperparameters on validation and then discarding it would
        throw away a week of data the model is entitled to learn from, so the
        final fit uses both. The test split is never included.
        """
        return self.series[self.train.start : self.validation.stop]

    @property
    def fit_times(self) -> np.ndarray:
        return self.times[self.train.start : self.validation.stop]

    def describe(self) -> str:
        return "\n".join(split.describe() for split in self)


def _day_index(times: np.ndarray) -> np.ndarray:
    """Local calendar date for each timestamp."""
    return times.astype("datetime64[D]")


def _bounds(times: np.ndarray, window: DateRange, name: str) -> tuple[int, int]:
    """First and one-past-last index of ``window`` within ``times``.

    Raises:
        SplitError: If the window is absent from the series.
    """
    days = _day_index(times)
    start_day = np.datetime64(window.start, "D")
    end_day = np.datetime64(window.end, "D")

    inside = np.flatnonzero((days >= start_day) & (days <= end_day))
    if inside.size == 0:
        raise SplitError(
            f"split {name} ({window}) does not overlap the series, which spans "
            f"{days[0]} to {days[-1]}"
        )
    start, stop = int(inside[0]), int(inside[-1]) + 1
    if stop - start != inside.size:
        raise SplitError(f"split {name} ({window}) is not contiguous in the series")
    return start, stop


def make_splits(
    values: np.ndarray,
    times: np.ndarray,
    config: Config,
    *,
    strict_lengths: bool = True,
) -> SplitSet:
    """Cut one area's series into the four configured splits.

    Args:
        values: The full series for one area.
        times: Local wall-clock timestamps, same length as ``values``.
        config: Study configuration supplying the split dates.
        strict_lengths: Assert each split has exactly the length its date range
            implies. Disable only for synthetic series in tests.

    Returns:
        A :class:`SplitSet` whose slices are views into ``values``.

    Raises:
        SplitError: On a length mismatch, a non-contiguous split, or a split
            absent from the series.
    """
    values = np.asarray(values)
    times = np.asarray(times)
    if values.shape[0] != times.shape[0]:
        raise SplitError(f"{values.shape[0]} values against {times.shape[0]} timestamps")

    per_day = config.dataset.daily_period
    built: dict[str, Split] = {}

    for name, window in config.splits.ordered:
        start, stop = _bounds(times, window, name)
        length = stop - start
        expected = window.n_points(per_day)
        if strict_lengths and length != expected:
            raise SplitError(
                f"split {name} ({window}) has {length} points, expected {expected}. "
                "The series does not match the configured boundaries."
            )
        built[name] = Split(
            name=name,
            values=values[start:stop],
            times=times[start:stop],
            start=start,
            stop=stop,
        )

    ordered = [built[name] for name, _ in config.splits.ordered]
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if later.start != earlier.stop:
            raise SplitError(
                f"split {later.name} starts at {later.start} but {earlier.name} "
                f"ends at {earlier.stop}; splits must be contiguous"
            )

    return SplitSet(
        train=built["train"],
        validation=built["validation"],
        test=built["test"],
        stress=built["stress"],
        series=values,
        times=times,
    )
