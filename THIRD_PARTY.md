# Third-party attribution

The repository's own code uses the authors' selected MIT license. This does not relicense third-party material.

**PosteriorDB.** Data, model specifications, metadata and reference posterior draws are from [stan-dev/posteriordb](https://github.com/stan-dev/posteriordb/tree/5545a1dd07ae297c36edecbcd82aa49097b4c385), revision `5545a1dd07ae297c36edecbcd82aa49097b4c385`. Its BSD 3-Clause copyright and terms are preserved in `data/posteriordb/LICENCE.md`. The bundled model-info records identify the models as BSD3. Cite the PosteriorDB paper and the original model sources identified in the metadata.

**SCS.** The cap projection, Jacobian and great-circle step-out algorithm are credited to the Sub-Cauchy sampler paper and [liusf15/sub-cauchy-sampler](https://github.com/liusf15/sub-cauchy-sampler/blob/25ea5ee21ae93a1737bbc34bd18a763f191f85de/src/scp_core.py), inspected at revision `25ea5ee21ae93a1737bbc34bd18a763f191f85de`. The repository uses a NumPy implementation derived from the supplied experiment code, with independent inverse/Jacobian/invariance checks. It is not represented as the authors' original performance implementation or an exact reproduction of their optimizer.

NumPy, SciPy, JAX, NumPyro, BlackJAX, ArviZ and Matplotlib are installed as dependencies and retain their own licenses. Cite relevant computational libraries when reporting the experiments.
