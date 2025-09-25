from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Int


class SaveConf(eqx.Module):
    initial: bool = True
    final: bool = True
    generation: bool = True
    generations: Int | None = None

    save_fitness: bool = True
    save_best_solutions: int | None = 1
    save_population: bool = True
    save_fitness_mean: bool = False
    save_fitness_var: bool = False
    save_population_diversity: bool = False

    # TODO: Determine how the input arguments of this function will look like.
    save_fn: Callable[..., Any] | None = None
