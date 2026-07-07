import numpy as np

class SphericalMesh:
    """ Spherical mesh object with configurable planetary radius. 
    
        Pass to FieldSet object as ``mesh=SphericalMesh(radius=...)``. 
        radius is in meters; None reverts degree to meter conversion 
        to 1852 * 60 ."""
    
    def __init__(self, radius: float | None = None):
        self.radius = radius

    @property
    def deg2m(self) -> float: 
        """ Meters per degree of arc."""
        if self.radius is None:
            return 1852 * 60.0
        else:
            return self.radius * np.pi / 180.0
        
    def __repr__(self) -> str:
        return f"SphericalMesh(radius={self.radius})"
    
    