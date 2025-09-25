from abc import abstractmethod
from typing import Generic, Protocol, TypeVar

from genetix.typing import IntScalarLike, RealScalarLike
from jaxtyping import PyTree

Args = TypeVar("Args", bound=PyTree)
Solution = TypeVar("Solution", bound=PyTree, contravariant=True)


class FitnessFunction(Protocol, Generic[Solution]):
    @abstractmethod
    @staticmethod
    def __call__(generation: IntScalarLike, solution: Solution) -> RealScalarLike:
        ...
