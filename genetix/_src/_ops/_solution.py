from collections.abc import Callable
from typing import Generic, TypeVar

import equinox as eqx
from genetix._src._genes import AbstractGene
from jaxtyping import PyTree

SolutionGenes = TypeVar("SolutionGenes", bound=PyTree[AbstractGene])
Phenotype = TypeVar("Phenotype", bound=PyTree, default=SolutionGenes)


def default_solution_factory(genes: PyTree) -> PyTree:
    return genes


class Species(eqx.Module, Generic[SolutionGenes, Phenotype]):
    genes: SolutionGenes
    phenotype_recipe: Callable[[SolutionGenes], Phenotype] = eqx.field(
        default_factory=lambda: default_solution_factory
    )
