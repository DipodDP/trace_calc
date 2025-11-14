import numpy as np
import pytest

from trace_calc.domain.models.coordinates import Coordinates
from trace_calc.application.services.coordinates import CoordinatesService
from trace_calc.domain.validators import ValidationError


def test_coordinates_service_with_numpy_array_raises_error():
    coord_a_np = np.array([59.585, 154.141139])
    coord_b_np = np.array([59.569594, 151.256373])

    with pytest.raises(ValidationError):
        CoordinatesService(coord_a_np, coord_b_np)


def test_coordinates_service_with_coordinates_object_does_not_raise_error():
    coord_a = Coordinates(lat=59.585, lon=154.141139)
    coord_b = Coordinates(lat=59.569594, lon=151.256373)

    try:
        CoordinatesService(coord_a, coord_b)
    except ValidationError:
        pytest.fail("ValidationError was raised unexpectedly")