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
from invrs_gym.challenges.diffract import metagrating_challenge, splitter_challenge
from invrs_gym.challenges.library import challenge as library_challenge
from invrs_gym.challenges.metalens import challenge as metalens_challenge
from invrs_gym.utils import metrics
from totypes import json_utils, types

from invrs_leaderboard import data

PyTree = Any


# Override the sim parameters for certain challenges. The default simulation params
# for many challenges are a tradeoff between simulation cost and accuracy. For eval
# purposes, we use settings that ensure greater accuracy. Note: if these values
# change, leaderboards will be affected and must be regenerated!
OVERRIDE_SIM_PARAMS_BY_CHALLENGE = {
    "diffractive_splitter": dataclasses.replace(
        splitter_challenge.DIFFRACTIVE_SPLITTER_SIM_PARAMS,
        approximate_num_terms=1000,
    ),
    "metagrating": dataclasses.replace(
        metagrating_challenge.METAGRATING_SIM_PARAMS,
        approximate_num_terms=400,
    ),
    "meta_atom_library": dataclasses.replace(
        library_challenge.LIBRARY_SIM_PARAMS,
        approximate_num_terms=300,
        wavelength=jnp.arange(0.45, 0.66, 0.02),
    ),
    "metalens": dataclasses.replace(
        metalens_challenge.METALENS_SIM_PARAMS,
        approximate_num_terms=400,
        num_layers=30,
    ),
}


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

    Raises:
        RuntimeError: If 64-bit jax is not enabled.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("64-bit must be enabled for eval calculations.")

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

    if challenge_name in OVERRIDE_SIM_PARAMS_BY_CHALLENGE:
        sim_params = OVERRIDE_SIM_PARAMS_BY_CHALLENGE[challenge_name]
        challenge = challenges.BY_NAME[challenge_name](  # type: ignore[operator]
            sim_params=sim_params
        )
    else:
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
            binarization_degree = metrics.binarization_degree(params=solution)
            results = dict(
                path=solution_path,
                eval_metric=data.try_float(eval_metric),
                minimum_width=data.try_float(minimum_width),
                minimum_spacing=data.try_float(minimum_spacing),
                binarization_degree=data.try_float(binarization_degree),
            )
            output_str = data.SEP.join(
                [f"{key}={value}" for key, value in results.items()]
            )
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
    leaderboard_paths = [entry[data.PATH] for entry in data.load_leaderboard().values()]
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


def get_solution_paths(base_path: str = "") -> List[str]:
    """Return the filenames for solutions for the given `base_path`."""
    base_path = data.fix_base_path(base_path)
    solution_paths = glob.glob(f"{base_path}challenges/*/solutions/*.csv")
    solution_paths += glob.glob(f"{base_path}challenges/*/solutions/*.json")
    return [path[len(base_path) :] for path in solution_paths]


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
        arrays = arrays > (leaf.lower_bound + leaf.upper_bound) / 2
        for arr in arrays:
            width, spacing = imageruler.minimum_length_scale(
                onp.asarray(arr), periodic=leaf.periodic
            )
            min_width = width if min_width is None else min(width, min_width)
            min_spacing = spacing if min_spacing is None else min(spacing, min_spacing)

    return min_width, min_spacing
