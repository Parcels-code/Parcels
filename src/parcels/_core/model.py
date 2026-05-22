from abc import ABC, abstractmethod
from typing import Any

import uxarray as ux
import xarray as xr

from parcels._core.basegrid import BaseGrid
from parcels._core.field import Field
from parcels._core.uxgrid import UxGrid
from parcels._core.xgrid import XGrid


class Model(ABC):
    data: Any
    grid: BaseGrid

    @abstractmethod
    def construct_fields(self) -> list[Field]: ...


class StructuredModel(Model):
    def __init__(self, data: xr.Dataset, grid: XGrid):
        self.data = data
        self.grid = grid

    def construct_fields(self) -> list[Field]: ...


class UnstructuredModel(Model):
    def __init__(self, data: ux.UxDataset, grid: UxGrid):
        self.data = data
        self.grid = grid

    def construct_fields(self) -> list[Field]: ...
