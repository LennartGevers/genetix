from abc import abstractmethod
from typing import Generic, Protocol, TypeVar

from genetix.typing import IntScalarLike, RealScalarLike
from jaxtyping import PyTree

Args = TypeVar("Args", bound=PyTree)
Phenotype = TypeVar("Phenotype", bound=PyTree, contravariant=True)


class FitnessFunction(Protocol, Generic[Phenotype]):
    @abstractmethod
    @staticmethod
    def __call__(generation: IntScalarLike, phenotype: Phenotype) -> RealScalarLike:
        ...
