"""Minimal heuristic-based shark probability estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Hotspot:
    name: str
    lat: float
    lon: float
    weight: float
    radius_km: float


HOTSPOTS: tuple[Hotspot, ...] = (
    Hotspot("Eastern Australia", -25.5, 153.0, 0.22, 750.0),
    Hotspot("South Africa", -34.0, 18.5, 0.18, 550.0),
    Hotspot("Hawaii", 20.9, -157.0, 0.14, 450.0),
    Hotspot("Florida", 27.5, -80.0, 0.16, 500.0),
    Hotspot("California", 34.0, -119.5, 0.12, 500.0),
    Hotspot("Brazil", -22.9, -43.2, 0.10, 420.0),
    Hotspot("Red Sea", 20.0, 38.5, 0.10, 350.0),
)


@dataclass(frozen=True)
class LandRegion:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    penalty: float


# Rough bounding boxes for continental interiors (used to dampen predictions on land).
LAND_INTERIORS: tuple[LandRegion, ...] = (
    LandRegion(25.0, 50.0, -110.0, -90.0, 0.35),  # Central USA
    LandRegion(-20.0, 5.0, -70.0, -50.0, 0.40),   # Amazon basin interior
    LandRegion(-10.0, 20.0, 10.0, 35.0, 0.45),    # Central Africa
    LandRegion(30.0, 60.0, 40.0, 120.0, 0.50),    # Eurasian interior
    LandRegion(-35.0, -15.0, 120.0, 135.0, 0.40), # Australian interior
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _wrap_longitude(lon: float) -> float:
    """Wrap longitude into the [-180, 180) range."""
    wrapped = (lon + 180.0) % 360.0 - 180.0
    # Handle the edge case where lon == 180 exactly.
    return wrapped if wrapped != -180.0 else 180.0


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate great-circle distance between two points on Earth."""
    radius_earth_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_earth_km * c


def _lat_factor(lat: float) -> float:
    """Favor latitudes near the tropics and temperate coastal waters."""
    # Cosine squared declines smoothly from equator towards poles.
    equatorial_bias = math.cos(math.radians(lat)) ** 2
    # Additional boost inside the tropics (|lat| <= 30°).
    tropic_span = clamp(1.0 - abs(lat) / 30.0)
    # Mild penalty for extreme poles.
    polar_penalty = clamp((abs(lat) - 50.0) / 40.0)
    return clamp(0.6 * equatorial_bias + 0.4 * tropic_span - 0.3 * polar_penalty, 0.0, 1.0)


def _current_factor(lat: float, lon: float) -> float:
    """Heuristic for warm currents and productive waters."""
    lon_rad = math.radians(_wrap_longitude(lon))
    lat_rad = math.radians(lat)
    # Combine sinusoidal patterns to emulate known warm currents.
    warm_current = 0.5 * (1.0 + math.sin(lon_rad * 1.5) * math.cos(lat_rad))
    upwelling = 0.5 * (1.0 + math.cos(lon_rad * 0.7 + math.sin(lat_rad)))
    return clamp(0.6 * warm_current + 0.4 * upwelling, 0.0, 1.0)


def _hotspot_contribution(lat: float, lon: float, hotspots: Iterable[Hotspot]) -> float:
    score = 0.0
    for spot in hotspots:
        distance = _haversine_distance_km(lat, lon, spot.lat, spot.lon)
        influence = math.exp(-(distance / spot.radius_km) ** 2)
        score += spot.weight * influence
    return score


def _land_penalty(lat: float, lon: float, regions: Iterable[LandRegion]) -> float:
    penalty = 0.0
    for region in regions:
        if region.lat_min <= lat <= region.lat_max and region.lon_min <= lon <= region.lon_max:
            penalty += region.penalty
    return penalty


def shark_probability(lat: float, lon: float) -> float:
    """Estimate the likelihood of encountering sharks at the given coordinates.

    The heuristic prioritizes tropical and temperate waters, emphasizes known
    global hotspots, and dampens probabilities over large continental interiors.

    Args:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).

    Returns:
        Probability between 0 and 1, rounded to three decimals.
    """
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("Latitude and longitude must be finite numbers.")

    lat = clamp(lat, -90.0, 90.0)
    lon = _wrap_longitude(lon)

    base = 0.05
    lat_component = 0.35 * _lat_factor(lat)
    current_component = 0.20 * _current_factor(lat, lon)
    hotspot_component = _hotspot_contribution(lat, lon, HOTSPOTS)
    land_penalty = _land_penalty(lat, lon, LAND_INTERIORS)

    probability = clamp(base + lat_component + current_component + hotspot_component - land_penalty)
    return round(probability, 3)


if __name__ == "__main__":
    print("Shark presence probability estimator")
    try:
        user_lat = float(input("Enter latitude (-90 to 90): ").strip())
        user_lon = float(input("Enter longitude (-180 to 180): ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    else:
        try:
            probability = shark_probability(user_lat, user_lon)
        except ValueError as exc:
            print(f"Error: {exc}")
        else:
            print(f"Estimated probability of encountering sharks: {probability:.3f}")
