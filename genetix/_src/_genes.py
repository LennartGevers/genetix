import abc

import equinox as eqx
import jax.numpy as jnp
from genetix._src._utils import linear_rescale
from genetix.typing import RealScalarLike
from jaxtyping import Bool, Int, Real


class AbstractGene(abc.ABC):
    @abc.abstractmethod
    def linear_rescale(self, value: Real) -> Real:
        ...

    @abc.abstractmethod
    def uniform(self, value: Real) -> Real:
        ...

    @abc.abstractmethod
    def compliant_value(self, value: Real) -> Bool:
        ...


class ContinuousGene(eqx.Module, AbstractGene):
    lower_bound: RealScalarLike = eqx.field(converter=jnp.asarray)
    upper_bound: RealScalarLike = eqx.field(converter=jnp.asarray)

    def __check_init__(self) -> None:
        if self.upper_bound < self.lower_bound:
            msg = (
                f"The Genes upper and lower bounds"
                f"({self.lower_bound}: {self.upper_bound}) are not in order."
            )
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"{self.__qualname__}({self.lower_bound}:{self.upper_bound})"

    @property
    def _interval_spread(self) -> RealScalarLike:
        return self.upper_bound - self.lower_bound

    def liner_rescale(self, value: Real) -> Real:
        return linear_rescale(self.lower_bound, value, self.upper_bound)

    def value_from_uniform(self, value: Real) -> Real:
        return self.lower_bound + value * self._interval_spread

    def compliant_value(self, value: Real) -> Real:
        # might need array broadcasting here? -> Or jnp.vectorize?
        return self.lower_bound < value & value < self.upper_bound


class DiscreteGene(eqx.Module, AbstractGene):
    values: Real | Int = eqx.field(converter=jnp.asarray)

    def linear_rescale(self, value: RealScalarLike | Int) -> RealScalarLike:
        return linear_rescale(jnp.min(self.values), value, jnp.max(self.values))

    def compliant_value(self, value: Real) -> Real:
        # might need array broadcasting here? -> Or jnp.vectorize?
        return jnp.isin(value, self.values)

    def value_from_uniform(self, value: Real) -> Real:
        diff = jnp.abs(self.values - value)

        index = jnp.argmin(diff, axis=-1)

        return self.values[index]

    def __repr__(self) -> str:
        return f"{self.__qualname__}({str(self.values)})"
