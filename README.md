# leaderboard

The leaderboard repo tracks solutions to [invrs-gym](https://github.com/invrs-io/gym) challenges.

## How to contribute your solutions

- Clone this repository
- Add your solution to the appropriate directory, following the filename convention i.e. `challenges/{CHALLENGE_NAME}/solutions/{DATE}_{GITHUB_USERNAME}_{SOLUTION_ID}.json`
- Modify the `README.md` in the challenge directory to include your submission. Please link any publication or code that is relevant.
- Run `python scripts/eval.py`, which will detect new designs and add them to the appropriate leaderboard files
- Submit a PR for review
