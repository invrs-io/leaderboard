# leaderboard

The leaderboard repo tracks solutions to [invrs-gym](https://github.com/invrs-io/gym) challenges.

Solutions are the `params` which are passed to the `response` method of the component associated with a gym challenge. They can be saved to json format by,

```python
from totypes import json_utils

serialized = json_utils.json_from_pytree(params)
with open("my_solution.json", "w") as f:
  f.write(serialized)
```

The evaluation metrics include the `eval_metric` scalar associated with each challenge, and the minimum width and spacing measured using the [imageruler](https://github.com/NanoComp/imageruler) algorithm.

## How to contribute your solutions

- Clone this repository
- Add your solution to the appropriate directory, following the filename convention (e.g. `challenges/{CHALLENGE_NAME}/solutions/{YYMMDD}_{GITHUB_USERNAME}_{SOLUTION_ID}.json`)
- Modify the `README.md` in the challenge directory to include your submission. Please link any publication or code that is relevant.
- Run `python scripts/eval.py`, which will detect new designs and add them to the appropriate leaderboard files
- Submit a PR for review
