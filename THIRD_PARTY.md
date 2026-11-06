# Third-party attribution

Project code is released under the MIT license. This does not relicense third-party material.

**Sub-Cauchy Sampler.** The 100D/500D comparisons call unmodified author projection and step-out transition code from [liusf15/sub-cauchy-sampler](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565), pinned at revision `ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`. Byte-for-byte source, checksums, Git blob identities and Sifan Liu's MIT notice are preserved in `third_party/scs_paper/`. The [paper](https://arxiv.org/abs/2601.11066) is by Sebastiano Grazzi, Sifan Liu, Gareth O. Roberts and Jun Yang.

Original SCS keeps the author-fitted map and h=0.1. The zero-centre/zero-observer control supplies different map parameters and R/h calibrated on that final map using the same pilot policy and budgets as DCS. This is a declared parameter modification, not a change to author projection or transition code. The public preparation driver reproduces fitting and calibration; ordinary sampling runs load the frozen inputs. See [the protocol](docs/REPORTED_STUDY.md).

The adapter removes progress callbacks, compiles before timing and records radial summaries in bounded scan blocks. The shared target is normal-CDF-skewed t(3), distinct from the SCS paper's Student-t-CDF targets. The Numba extension is labelled as an implementation comparison.

Every VI-enabled SCS fit uses the pinned original optimizer. Logistic and super-heavy-tail production retain checked NumPy/Numba implementations derived from supplied experiments and inspected against upstream `25ea5ee21ae93a1737bbc34bd18a763f191f85de`; they are not represented as unmodified author performance implementations. See [source details](third_party/scs_paper/README.md).

**PosteriorDB.** Data, models, metadata and reference draws come from [stan-dev/posteriordb](https://github.com/stan-dev/posteriordb/tree/5545a1dd07ae297c36edecbcd82aa49097b4c385), revision `5545a1dd07ae297c36edecbcd82aa49097b4c385`. BSD 3-Clause terms are preserved in `data/posteriordb/LICENCE.md`. Cite PosteriorDB and the original models identified in the metadata.

NumPy, SciPy, JAX, NumPyro, BlackJAX, ArviZ, Numba and Matplotlib retain their own licenses. Cite the computational libraries used when reporting results.
