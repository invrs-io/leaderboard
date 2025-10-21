# Leaderboard

The leaderboard is a dataset of user-generated solutions to optical/photonic design challenges from the [invrs-gym](https://invrs-io.github.io/gym/index.html). The leaderboard tracks key quantities for each solution related to the _quality_ and _degree of manufacturability_:
- Performance metric: this is an eval metric defined uniquely for each challenge. This may be a physically meaningful quantity (such as efficiency), or a synthetic quantity that balances multiple objectives. A higher eval metric corresponds to a more desirable solution.
- Minimum length scale: the minimum width and spacing of features in each design are measured using the [imageruler](https://github.com/NanoComp/imageruler) package. The smaller of these two is the minimum length scale. In general, designs with larger minimum length scales are more easily manufactured.

High-quality, highly-manufacturable solutions to gym challenges can be of practical value. Many gym challenges are based on real-world photonic/optical design challenges---in areas such as color-filter-free imaging, large-area metalenses, and quantum information processing---and solutions can be directly fabricated. In fact, many solutions to gym challenges have already been manufactured and reported in the scientific literature.

| ![Meta-atom library designs](https://github.com/invrs-io/leaderboard/blob/main/docs/img/meta_atom_library_design.png?raw=true) |
|:--:|
| *Solution to the [meta-atom library challenge](https://invrs-io.github.io/leaderboard/notebooks/challenges/meta_atom_library.html).* |

Therefore, _algorithms_ which reliably obtain high-quality, manufacturable solutions to gym challenges are valuable. The joint purpose of the invrs-gym and leaderboard are to aid and accelerate the development of such algorithms.
