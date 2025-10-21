"""Verification tests for new solutions added to the repo.

Copyright (c) 2025 invrs.io LLC
"""

import unittest

from invrs_leaderboard import summary


class SummaryTest(unittest.TestCase):
    def test_summary_matches_expected(self):
        expected_summary = summary.generate_summary()
        with open("challenges/README.md") as f:
            actual_summary = f.read()
        self.assertEqual(actual_summary, expected_summary)
