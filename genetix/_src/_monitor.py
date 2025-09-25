import abc
from collections.abc import Generator
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import PyTree

_InternalState = TypeVar("_InternalState", bound=PyTree)


class AbstractMonitor(eqx.Module, Generic[_InternalState]):
    @abc.abstractmethod
    def init(self) -> Generator[_InternalState, None, None]:
        ...

    @abc.abstractmethod
    def update(self, state: PyTree, internal_state: _InternalState) -> _InternalState:
        ...


class NoMonitor(AbstractMonitor[None]):
    def init(self) -> Generator[None, None, None]:
        yield None

    def update(self, state: PyTree, internal_state: None) -> None:
        return internal_state


class TextMonitor(AbstractMonitor[None]):
    def init(self) -> Generator[None, None, None]:
        yield None

    def update(self, state: PyTree, internal_state: None) -> None:
        print(state)
        return internal_state
