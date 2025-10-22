# Submissions

This page details the process of submitting a solution to the leaderbaord.

## Save your solution: json format

If you directly used the invrs-gym in generating your solution, you should have a `params` pytree available containing `Density2DArray` objects and perhaps other parameters. This can be saved to json format by,

```python
from totypes import json_utils

serialized = json_utils.json_from_pytree(params)
with open("my_solution.json", "w") as f:
  f.write(serialized)
```

## Save your solution: csv format

If you generated solutions using your own implementation of a gym challenge, you can still submit solutions to most challenges without manually generating a `params` pytree. Simply save the density array as a csv file using your method of choice. Important: this method only works for challenges where the design variables contain a single density array. Also, when your solution is evaluated, default values are used for design variables other than the density.

## How to contribute your solutions

- Open source your solutions on your own GitHub repo, using MIT or similar license.
- Clone this repository.
- Install via `pip install -e .`
- Add your solution to the appropriate directory, following the filename convention `challenges/{CHALLENGE_NAME}/solutions/{YYMMDD}_{GITHUB_USERNAME}_{ID}.json`. The `ID` may be any meaningful string of your choosing.
- Modify the `README.md` in the challenge directory to include your submission. Please link your GitHub repo, and feel free to link and any publication or code that is relevant.
- Run `python scripts/eval.py`, which will detect new designs and add them to the appropriate leaderboard files.
- Run `python scripts/summary.py` which will regenerate the `challenges/README.md` including your submissions. Take care that the `README.md` associated with each challenge has been properly updated.
- Submit a PR for review. A github action will re-run the evaluation of your designs and ensure they match the leaderboard updates.
