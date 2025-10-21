"""Verification tests for new solutions added to the repo.

Copyright (c) 2025 invrs.io LLC
"""

import functools
import pytest
import tempfile
import unittest

import git
import jax
import numpy as onp
from invrs_gym import challenges

from invrs_leaderboard import data, utils

jax.config.update("jax_enable_x64", True)


@functools.lru_cache(maxsize=None)
def _load_repo_leaderboard():
    """Loads the leaderboard from the main branch on github."""
    with tempfile.TemporaryDirectory() as temp_path:
        repo_url = "https://github.com/invrs-io/leaderboard.git"
        git.Repo.clone_from(repo_url, temp_path, branch="main")
        return data.load_leaderboard(base_path=temp_path)


def _get_new_leaderboard_entries():
    """Get all new or updated leaderboard entries."""
    local_leaderboard = data.load_leaderboard()
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


# Use a tighter tolerance for evaluation metric.
RTOL_EVAL_METRIC = 1e-5
RTOL_OTHER = 1e-3


class VerifyLeaderboardTest(unittest.TestCase):
    def test_all_solutions_on_leaderboard(self):
        """Check that all solution files have corresponding leaderboard entries."""
        solution_paths = utils.get_solution_paths()
        leaderboard_paths = [
            entry[data.PATH] for entry in data.load_leaderboard().values()
        ]
        self.assertEqual(len(solution_paths), len(leaderboard_paths))
        self.assertSetEqual(set(leaderboard_paths), set(solution_paths))

    def _test_new_submissions_have_correct_metrics(
        self,
        challenge_to_check,
        disable_jit=False,
    ):
        new_leaderboard_entries = _get_new_leaderboard_entries()

        new_leaderboard_entries_by_challenge = {}
        for entry in new_leaderboard_entries.values():
            challenge_name = entry[data.PATH].split("/")[1]
            if challenge_name != challenge_to_check:
                continue
            assert challenge_name in challenges.BY_NAME.keys()
            if challenge_name not in new_leaderboard_entries_by_challenge:
                new_leaderboard_entries_by_challenge[challenge_name] = []
            new_leaderboard_entries_by_challenge[challenge_name].append(entry)

        for challenge_name, entries in new_leaderboard_entries_by_challenge.items():
            evaluation_results = utils.evaluate_solutions_to_challenge(
                challenge_name=challenge_name,
                solution_paths=[entry[data.PATH] for entry in entries],
                update_leaderboard=False,
                print_results=True,
                disable_jit=disable_jit,
            )

            for solution_path in evaluation_results.keys():
                reported = new_leaderboard_entries[solution_path]
                expected = evaluation_results[solution_path]
                for key in reported.keys():
                    with self.subTest(f"{solution_path=}/{key=}"):
                        a = data.try_float(reported[key])
                        b = data.try_float(expected[key])
                        if isinstance(a, float):
                            rtol = (
                                RTOL_EVAL_METRIC if key == "eval_metric" else RTOL_OTHER
                            )
                            onp.testing.assert_allclose(a, b, rtol=rtol)
                        else:
                            self.assertEqual(a, b)

    @pytest.mark.slow
    def test_bayer_sorter(self):
        self._test_new_submissions_have_correct_metrics("bayer_sorter")

    @pytest.mark.slow
    def test_ceviche_beam_splitter(self):
        self._test_new_submissions_have_correct_metrics("ceviche_beam_splitter")

    @pytest.mark.slow
    def test_ceviche_mode_converter(self):
        self._test_new_submissions_have_correct_metrics("ceviche_mode_converter")

    @pytest.mark.slow
    def test_ceviche_power_splitter(self):
        self._test_new_submissions_have_correct_metrics("ceviche_power_splitter")

    @pytest.mark.slow
    def test_ceviche_waveguide_bend(self):
        self._test_new_submissions_have_correct_metrics("ceviche_waveguide_bend")

    @pytest.mark.slow
    def test_ceviche_wdm(self):
        self._test_new_submissions_have_correct_metrics("ceviche_wdm")

    @pytest.mark.slow
    def test_ceviche_lightweight_beam_splitter(self):
        self._test_new_submissions_have_correct_metrics(
            "ceviche_lightweight_beam_splitter"
        )

    @pytest.mark.slow
    def test_ceviche_lightweight_mode_converter(self):
        self._test_new_submissions_have_correct_metrics(
            "ceviche_lightweight_mode_converter"
        )

    @pytest.mark.slow
    def test_ceviche_lightweight_power_splitter(self):
        self._test_new_submissions_have_correct_metrics(
            "ceviche_lightweight_power_splitter"
        )

    @pytest.mark.slow
    def test_ceviche_lightweight_waveguide_bend(self):
        self._test_new_submissions_have_correct_metrics(
            "ceviche_lightweight_waveguide_bend"
        )

    @pytest.mark.slow
    def test_ceviche_lightweight_wdm(self):
        self._test_new_submissions_have_correct_metrics("ceviche_lightweight_wdm")

    @pytest.mark.slow
    def test_diffractive_splitter(self):
        self._test_new_submissions_have_correct_metrics("diffractive_splitter")

    @pytest.mark.slow
    def test_meta_atom_library(self):
        self._test_new_submissions_have_correct_metrics("meta_atom_library")

    @pytest.mark.slow
    def test_metagrating(self):
        self._test_new_submissions_have_correct_metrics("metagrating")

    @pytest.mark.slow
    def test_metalens(self):
        self._test_new_submissions_have_correct_metrics("metalens")

    @pytest.mark.slow
    def test_photon_extractor(self):
        self._test_new_submissions_have_correct_metrics(
            "photon_extractor", disable_jit=True
        )
