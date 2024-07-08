"""Evaluate solutions to the invrs-gym challenges and add them to the leaderboard.

Copyright (c) 2024 The INVRS-IO authors.
"""

import dataclasses
import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageruler
import jax
import jax.numpy as jnp
import numpy as onp
from invrs_gym import challenges
from totypes import json_utils, types

# 64 bit mode ensures highest accuracy for evaluation results.
jax.config.update("jax_enable_x64", True)

PyTree = Any

SEP = ", "
PATH = "path"


def update_leaderboard_with_new_solutions() -> None:
    """Updates the leaderboard with newly-added challenge solutions."""
    solution_paths = get_new_solution_paths()

    solution_paths_by_challenge: Dict[str, List[str]] = {}
    for path in solution_paths:
        challenge_name = path.split("/")[1]
        assert challenge_name in challenges.BY_NAME.keys()
        if challenge_name not in solution_paths_by_challenge:
            solution_paths_by_challenge[challenge_name] = []
        solution_paths_by_challenge[challenge_name].append(path)

    for solution_paths in solution_paths_by_challenge.values():
        solution_paths.sort()

    for challenge_name, paths in solution_paths_by_challenge.items():
        evaluate_solutions_to_challenge(
            challenge_name=challenge_name,
            solution_paths=paths,
            update_leaderboard=True,
            print_results=True,
        )


def evaluate_solutions_to_challenge(
    challenge_name: str,
    solution_paths: Sequence[str],
    update_leaderboard: bool,
    print_results: bool,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates given solutions to the specified challenge.

    Optionally, the leaderboard is updated and results are printed. Evaluation is
    performed on CPU, and must be done with 64-bit precision enabled.

    Args:
        challenge_name: The name of the challenge for which new solutions are to be
            evaluated.
        solution_paths: The paths to solutions to be evaluated.
        update_leaderboard: If `True`, the leaderboard is updated with the new
            evaluation results.
        print_results: If `True`, prints the evaluation results for each solution.

    Returns:
        A dict containing the evaluation results, with keys being the solution path.
    """
    for path in solution_paths:
        if not path.startswith(f"challenges/{challenge_name}/solutions/"):
            raise ValueError(
                f"Unexpected solution path for {challenge_name} challenge, "
                f"got `{path}`."
            )

    leaderboard_path = f"challenges/{challenge_name}/leaderboard.txt"
    if update_leaderboard and not os.path.exists(leaderboard_path):
        with open(leaderboard_path, "w") as f:
            f.write("")

    challenge = challenges.BY_NAME[challenge_name]()  # type: ignore[operator]
    example_solution = challenge.component.init(jax.random.PRNGKey(0))

    solutions = {}
    for path in solution_paths:
        solutions[path] = load_solution(path, example_solution=example_solution)

    evaluation_results = {}
    with jax.default_device(jax.devices("cpu")[0]):

        @jax.jit
        def eval_metric_fn(params):
            response, _ = challenge.component.response(params)
            return challenge.eval_metric(response)

        for solution_path, solution in solutions.items():
            eval_metric = float(eval_metric_fn(params=solution))
            minimum_width, minimum_spacing = compute_length_scale(solution)
            results = dict(
                path=solution_path,
                eval_metric=_try_float(eval_metric),
                minimum_width=_try_float(minimum_width),
                minimum_spacing=_try_float(minimum_spacing),
            )
            output_str = SEP.join([f"{key}={value}" for key, value in results.items()])
            evaluation_results[solution_path] = results
            if print_results:
                print(output_str)
            if update_leaderboard:
                with open(leaderboard_path, "a") as f:
                    f.write(output_str)
                    f.write("\n")

    return evaluation_results


def get_new_solution_paths() -> List[str]:
    """Returns paths to new solutions for all challenges."""
    solution_paths = get_solution_paths()
    leaderboard_paths = [entry[PATH] for entry in load_leaderboard().values()]
    return [path for path in solution_paths if path not in leaderboard_paths]


def load_solution(path: str, example_solution: PyTree) -> PyTree:
    """Loads solutions in `csv` or `json` format."""

    if path.endswith(".json"):
        with open(path) as f:
            serialized_solution = f.read()
        solution = json_utils.pytree_from_json(serialized_solution)
    elif path.endswith(".csv"):
        density_array = onp.genfromtxt(path, delimiter=",")
        # If the solution is in a csv file, it should contain a single density
        # array. Replace the density array of the example solution with the
        # the loaded density array, retaining all the other default variables.
        solution = jax.tree_util.tree_map(
            lambda x: (
                dataclasses.replace(
                    x, array=density_array, fixed_solid=None, fixed_void=None
                )
                if isinstance(x, types.Density2DArray)
                else x
            ),
            example_solution,
            is_leaf=lambda x: isinstance(x, types.Density2DArray),
        )

    # Ensure that there are no weak types.
    solution = jax.tree_util.tree_map(lambda x: jnp.array(x, x.dtype), solution)
    return solution


def load_leaderboard(base_path: str = "") -> Dict[str, Any]:
    """Load leaderboard data for the given `base_path`."""
    base_path = _fix_base_path(base_path)
    leaderboard_paths = glob.glob(f"{base_path}challenges/*/leaderboard.txt")
    leaderboard = {}

    for path in leaderboard_paths:
        with open(path) as f:
            leaderboard_data = f.read().strip()
        if not leaderboard_data:
            continue
        lines = leaderboard_data.split("\n")
        for line in lines:
            entry_dict = {}
            for d in line.split(SEP):
                key, value = d.split("=")
                entry_dict[key] = _try_float(value)
            solution_path = entry_dict[PATH]
            if solution_path in leaderboard:
                raise ValueError(
                    f"Found duplicate entry in leaderboard: {solution_path}"
                )
            leaderboard[solution_path] = entry_dict

    return leaderboard


def get_solution_paths(base_path: str = "") -> List[str]:
    """Return the filenames for solutions for the given `base_path`."""
    base_path = _fix_base_path(base_path)
    solution_paths = glob.glob(f"{base_path}challenges/*/solutions/*.csv")
    solution_paths += glob.glob(f"{base_path}challenges/*/solutions/*.json")
    return [path[len(base_path) :] for path in solution_paths]


def _fix_base_path(base_path: str) -> str:
    """Prepend the base path, if it is not empty."""
    if not base_path or base_path.endswith("/"):
        return base_path
    return f"{base_path}/"


def _try_float(value: Any) -> Any:
    """Convert `value` to float if possible."""
    try:
        return float(value)
    except ValueError:
        return value


def is_density(leaf: Any) -> bool:
    """Return `True` if `leaf` is a `Density2DArray`."""
    return isinstance(leaf, types.Density2DArray)


def compute_length_scale(params: Any) -> Tuple[Optional[int], Optional[int]]:
    """Compute minimum length scale in pixels for any density arrays in `params`."""
    min_width: Optional[int] = None
    min_spacing: Optional[int] = None
    for leaf in jax.tree_util.tree_leaves(params, is_leaf=is_density):
        if not is_density(leaf):
            continue
        arrays = leaf.array.reshape((-1,) + leaf.shape[-2:])
        arrays = arrays > (leaf.lower_bound + leaf.lower_bound) / 2
        for arr in arrays:
            width, spacing = imageruler.minimum_length_scale(onp.asarray(arr))
            min_width = width if min_width is None else min(width, min_width)
            min_spacing = spacing if min_spacing is None else min(spacing, min_spacing)

    return min_width, min_spacing
