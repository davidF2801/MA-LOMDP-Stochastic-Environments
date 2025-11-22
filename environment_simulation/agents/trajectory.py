"""Trajectory utilities for satellite-like agents."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Trajectory:
    """Discrete trajectory parameterized by latitude/longitude samples in degrees."""

    latitudes: np.ndarray
    longitudes: np.ndarray

    def __post_init__(self):
        self.latitudes = np.asarray(self.latitudes, dtype=float)
        self.longitudes = np.asarray(self.longitudes, dtype=float) % 360.0
        assert self.latitudes.shape == self.longitudes.shape, "Latitude and longitude arrays must align."
        assert self.latitudes.ndim == 1, "Trajectory arrays must be 1-D."
        self.period = int(self.latitudes.size)
        assert self.period > 0, "Trajectory must contain at least one waypoint."

    def position_at(self, step: int) -> tuple[float, float]:
        """Return (lat, lon) at a given simulation step."""
        idx = int(step) % self.period
        return float(self.latitudes[idx]), float(self.longitudes[idx])

    @classmethod
    def circular_orbit(
        cls,
        period: int,
        inclination_deg: float,
        phase_deg: float = 0.0,
        latitude_offset: float = 0.0,
    ) -> "Trajectory":
        """Generate a simple circular orbit-like trajectory."""
        steps = np.arange(period, dtype=float)
        theta = 2.0 * np.pi * steps / period
        lon = (phase_deg + 360.0 * steps / period) % 360.0
        lat = latitude_offset + inclination_deg * np.sin(theta)
        lat = np.clip(lat, -90.0, 90.0)
        return cls(latitudes=lat, longitudes=lon)

