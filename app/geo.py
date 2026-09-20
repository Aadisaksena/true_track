"""geo.py — small helper to work in local meters instead of lat/lon degrees."""
import math

EARTH_RADIUS_M = 6371000.0


class LocalFrame:
    """Equirectangular projection centered on a reference point. Good enough
    over the few-km scale of a single GNSS blackout; not for long-range use."""

    def __init__(self, ref_lat_deg: float, ref_lon_deg: float):
        self.ref_lat = math.radians(ref_lat_deg)
        self.ref_lon = math.radians(ref_lon_deg)
        self.cos_ref_lat = math.cos(self.ref_lat)

    def to_xy(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        x = (lon - self.ref_lon) * self.cos_ref_lat * EARTH_RADIUS_M  # east
        y = (lat - self.ref_lat) * EARTH_RADIUS_M  # north
        return x, y

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        lat = self.ref_lat + y / EARTH_RADIUS_M
        lon = self.ref_lon + x / (EARTH_RADIUS_M * self.cos_ref_lat)
        return math.degrees(lat), math.degrees(lon)
