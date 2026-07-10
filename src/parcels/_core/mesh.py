import numpy as np


class SphericalMesh:
    """Spherical mesh object with configurable planetary radius.

    Pass to FieldSet object as ``mesh=SphericalMesh(radius=...)``.
    radius is in meters; None reverts to default for Earth, where 
    arcdegree to meter conversion is defined as 1852 * 60
    (1852 meters per arcminute * 60 arcminutes per arcdegree).
    """

    def __init__(self, radius: float | None = None):
        if radius is not None and not isinstance(radius, (int, float, np.number)):
            raise TypeError(f"radius must be a number or None, got {type(radius).__name__}")
        if radius is not None and radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = radius

    @property
    def deg2m(self) -> float:
        """Meters per degree of arc."""
        if self.radius is None:
            return 1852 * 60.0
        else:
            return self.radius * np.pi / 180.0

    def __repr__(self) -> str:
        return f"SphericalMesh(radius={self.radius})"
