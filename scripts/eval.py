"""Evaluate solutions to the invrs-gym challenges and add them to the leaderboard.

Copyright (c) 2025 invrs.io LLC
"""

import jax

from invrs_leaderboard import utils

# 64 bit mode ensures highest accuracy for evaluation results.
jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    utils.update_leaderboard_with_new_solutions()
