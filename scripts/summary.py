"""Generate a summary of all leaderboard submissions.

Copyright (c) 2025 invrs.io LLC
"""

from invrs_leaderboard import summary


if __name__ == "__main__":
    text = summary.generate_summary()

    with open("challenges/README.md", "w") as f:
        f.write(text)
