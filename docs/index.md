# Overview

The leaderboard is a database of user-generated solutions to optical/photonic design challenges from the [invrs-gym](https://invrs-io.github.io/gym/index.html). The leaderboard tracks key quantities for each solution related to the _quality_ and _degree of manufacturability_:
- Performance metric: this is an eval metric defined uniquely for each challenge. This may be a physically meaningful quantity (such as efficiency), or a synthetic quantity that balances multiple objectives. A higher eval metric corresponds to a more desirable solution.
- Minimum length scale: the minimum width and spacing of features in each design are measured using the [imageruler](https://github.com/NanoComp/imageruler) package. The smaller of these two is the minimum length scale. In general, designs with larger minimum length scales are more easily manufactured.

High-quality, highly-manufacturable solutions to gym challenges can be of practical value. Many gym challenges are based on real-world photonic/optical design challenges--in areas such as color-filter-free imaging, large-area metalenses, and quantum information processing--and solutions can be directly fabricated. In fact, many solutions on the leaderboard have already been manufactured.

| ![Meta-atom library designs](/img/meta_atom_library_designs.png) |
|:--:|
| *Solutions to the [meta-atom library challenge](https://invrs-io.github.io/leaderboard/notebooks/meta_atom_library.html) from (a)-(c) “[Dispersion-engineered metasurfaces reaching broadband 90% relative diffraction efficiency](https://www.nature.com/articles/s41467-023-38185-2)” by Chen et al. and (d) submitted by @mfschubert. The right panel is a SEM image of a fabricated metagrating using design (a).* |

Therefore, _algorithms_ which reliably obtain high-quality, manufacturable soltutions to gym challenges are valuable. The joint purpose of the invrs-gym and leaderboard are to aid and accelerate the development of such algorithms.
