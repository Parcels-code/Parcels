import numpy as np

EARTH_RADIUS = 6366707.019493707


class SphericalMesh:
    """Spherical mesh object with configurable planetary radius.

    Pass to FieldSet object as ``mesh=SphericalMesh(radius=...)``.
    radius is in meters; defaults to Earth radius.
    """

    def __init__(self, radius: float = EARTH_RADIUS):
        if not isinstance(radius, (int, float, np.number)):
            raise TypeError(f"radius must be a number, got {type(radius).__name__}")
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = radius

    @property
    def deg2m(self) -> float:
        """Meters per degree of arc."""
        return self.radius * np.pi / 180.0

    def __repr__(self) -> str:
        return f"SphericalMesh(radius={self.radius})"
