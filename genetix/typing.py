from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    Real,
)

if TYPE_CHECKING:
    BoolScalarLike = bool | Array | np.ndarray
    FloatScalarLike = float | Array | np.ndarray
    IntScalarLike = int | Array | np.ndarray
    RealScalarLike = bool | int | float | Array | np.ndarray
else:
    BoolScalarLike = Bool[ArrayLike, ""]
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]
    RealScalarLike = Real[ArrayLike, ""]

RealInterval = tuple[RealScalarLike, RealScalarLike]
