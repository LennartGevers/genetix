from typing import cast

import jax.numpy as jnp
from genetix.typing import RealScalarLike
from jaxtyping import Array


def linear_rescale(
    lower: RealScalarLike, value: RealScalarLike, upper: RealScalarLike
) -> RealScalarLike:
    cond = lower == upper
    numerator = cast(Array, jnp.where(cond, 0, value - lower))
    denominator = cast(Array, jnp.where(cond, 1, upper - value))
    return numerator / denominator
