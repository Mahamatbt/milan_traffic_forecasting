"""Italian and Milanese public holidays falling in the observation period.

Traffic on a public holiday behaves like a Sunday regardless of the weekday it
lands on, so a model given only day-of-week has no way to anticipate it. These
dates are used twice: to cross-reference detected anomalies during exploration,
and as a calendar feature during modelling.

Sant'Ambrogio (7 December) is specific to Milan -- it is the city's patron
saint's day and a municipal holiday, and it coincides with the opening night of
the La Scala season. A national holiday calendar would miss it, and it falls
directly inside this dataset.

Sources: Italian national holidays (festività nazionali) and the Milan
municipal calendar for 2013.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

__all__ = ["Holiday", "HOLIDAYS", "holiday_dates", "is_holiday", "holiday_flags", "describe"]


@dataclass(frozen=True)
class Holiday:
    """One non-working day, with enough context to interpret a traffic anomaly."""

    date: dt.date
    name: str
    scope: str  # "national", "milan", or "observance"
    note: str = ""

    @property
    def weekday(self) -> str:
        return self.date.strftime("%A")


# Restricted to 2013-11-01 .. 2014-01-01, the observation period.
HOLIDAYS: tuple[Holiday, ...] = (
    Holiday(
        dt.date(2013, 11, 1),
        "Ognissanti (All Saints)",
        "national",
        "First day of the dataset, and a public holiday falling on a Friday.",
    ),
    Holiday(
        dt.date(2013, 12, 7),
        "Sant'Ambrogio",
        "milan",
        "Milan's patron saint; municipal holiday and the La Scala opening night.",
    ),
    Holiday(
        dt.date(2013, 12, 8),
        "Immacolata Concezione",
        "national",
        "Falls the day after Sant'Ambrogio, making a long weekend in Milan.",
    ),
    Holiday(
        dt.date(2013, 12, 24),
        "Vigilia di Natale (Christmas Eve)",
        "observance",
        "Not an official holiday, but most activity stops from the afternoon.",
    ),
    Holiday(dt.date(2013, 12, 25), "Natale (Christmas Day)", "national"),
    Holiday(dt.date(2013, 12, 26), "Santo Stefano (St Stephen's Day)", "national"),
    Holiday(
        dt.date(2013, 12, 31),
        "San Silvestro (New Year's Eve)",
        "observance",
        "Not an official holiday; traffic peaks around midnight rather than stopping.",
    ),
    Holiday(
        dt.date(2014, 1, 1),
        "Capodanno (New Year's Day)",
        "national",
        "Final day of the dataset.",
    ),
)


def holiday_dates(*, include_observances: bool = True) -> set[dt.date]:
    """The set of holiday dates.

    Args:
        include_observances: Include days that are not official holidays but on
            which activity clearly departs from normal, such as Christmas Eve.

    Returns:
        Dates as a set for fast membership tests.
    """
    return {h.date for h in HOLIDAYS if include_observances or h.scope != "observance"}


def is_holiday(day: dt.date, *, include_observances: bool = True) -> bool:
    """Whether a date is a holiday in Milan."""
    return day in holiday_dates(include_observances=include_observances)


def holiday_flags(timestamps: np.ndarray, *, include_observances: bool = True) -> np.ndarray:
    """Boolean flag per timestamp, for use as a model feature.

    Args:
        timestamps: Local wall-clock ``datetime64`` values.
        include_observances: See :func:`holiday_dates`.

    Returns:
        A boolean array the same length as ``timestamps``.
    """
    dates = holiday_dates(include_observances=include_observances)
    days = np.asarray(timestamps).astype("datetime64[D]").astype(dt.date)
    return np.array([day in dates for day in days], dtype=bool)


def describe(day: dt.date) -> Holiday | None:
    """Return the holiday falling on ``day``, or None."""
    for holiday in HOLIDAYS:
        if holiday.date == day:
            return holiday
    return None
