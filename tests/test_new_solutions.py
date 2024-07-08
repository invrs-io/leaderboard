"""Verification tests for new solutions added to the repo.

Copyright (c) 2024 The INVRS-IO authors.
"""

import functools
import tempfile
import unittest

import git
import jax
import numpy as onp
from invrs_gym import challenges

from invrs_leaderboard import utils

jax.config.update("jax_enable_x64", True)


@functools.lru_cache(maxsize=None)
def _load_repo_leaderboard():
    """Loads the leaderboard from the main branch on github."""
    with tempfile.TemporaryDirectory() as temp_path:
        repo_url = "https://github.com/invrs-io/leaderboard.git"
        git.Repo.clone_from(repo_url, temp_path, branch="main")
        return utils.load_leaderboard(base_path=temp_path)


def _get_new_leaderboard_entries():
    """Get all new or updated leaderboard entries."""
    local_leaderboard = utils.load_leaderboard()
    repo_leaderboard = _load_repo_leaderboard()
    new_entries = {}
    for key in local_leaderboard.keys():
        if key not in repo_leaderboard or not _dicts_identical(
            local_leaderboard[key], repo_leaderboard[key]
        ):
            new_entries[key] = local_leaderboard[key]
    return new_entries


def _dicts_identical(a, b):
    """Return `True` if `a` and `b` are identical."""
    if set(a.keys()) != set(b.keys()):
        return False
    for key in a.keys():
        if a[key] != b[key]:
            return False
    return True


class VerifyLeaderboardTest(unittest.TestCase):
    def test_all_solutions_on_leaderboard(self):
        """Check that all solution files have corresponding leaderboard entries."""
        solution_paths = utils.get_solution_paths()
        leaderboard_paths = [
            entry[utils.PATH] for entry in utils.load_leaderboard().values()
        ]
        self.assertEqual(len(solution_paths), len(leaderboard_paths))
        self.assertSetEqual(set(leaderboard_paths), set(solution_paths))

    def test_new_submissions_have_correct_metrics(self):
        new_leaderboard_entries = _get_new_leaderboard_entries()

        new_leaderboard_entries_by_challenge = {}
        for entry in new_leaderboard_entries.values():
            challenge_name = entry[utils.PATH].split("/")[1]
            assert challenge_name in challenges.BY_NAME.keys()
            if challenge_name not in new_leaderboard_entries_by_challenge:
                new_leaderboard_entries_by_challenge[challenge_name] = []
            new_leaderboard_entries_by_challenge[challenge_name].append(entry)

        for challenge_name, entries in new_leaderboard_entries_by_challenge.items():
            evaluation_results = utils.evaluate_solutions_to_challenge(
                challenge_name=challenge_name,
                solution_paths=[entry[utils.PATH] for entry in entries],
                update_leaderboard=False,
                print_results=False,
            )

            for solution_path in evaluation_results.keys():
                reported = new_leaderboard_entries[solution_path]
                expected = evaluation_results[solution_path]
                for key in reported.keys():
                    with self.subTest(f"{solution_path=}/{key=}"):
                        a = utils._try_float(reported[key])
                        b = utils._try_float(expected[key])
                        if isinstance(a, float):
                            onp.testing.assert_allclose(a, b)
                        else:
                            self.assertEqual(a, b)
