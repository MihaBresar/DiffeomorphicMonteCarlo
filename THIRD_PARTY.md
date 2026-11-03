# Third-party attribution

The repository's own code uses the authors' selected MIT license. This does not relicense third-party material.

**PosteriorDB.** Data, model specifications, metadata and reference posterior draws are from [stan-dev/posteriordb](https://github.com/stan-dev/posteriordb/tree/5545a1dd07ae297c36edecbcd82aa49097b4c385), revision `5545a1dd07ae297c36edecbcd82aa49097b4c385`. Its BSD 3-Clause copyright and terms are preserved in `data/posteriordb/LICENCE.md`. The bundled model-info records identify the models as BSD3. Cite the PosteriorDB paper and the original model sources identified in the metadata.

**SCS.** The cap projection, Jacobian and great-circle step-out algorithm are credited to the Sub-Cauchy sampler paper and [liusf15/sub-cauchy-sampler](https://github.com/liusf15/sub-cauchy-sampler/blob/25ea5ee21ae93a1737bbc34bd18a763f191f85de/src/scp_core.py), inspected at revision `25ea5ee21ae93a1737bbc34bd18a763f191f85de`. The repository uses a NumPy implementation derived from the supplied experiment code, with independent inverse/Jacobian/invariance checks. It is not represented as the authors' original performance implementation or an exact reproduction of their optimizer.

The single 100D skewed ν=3 paper comparison separately uses the original author
SCS code at paper-date revision
[`ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565).
Byte-for-byte source copies, checksums and Sifan Liu's MIT licence are preserved
in `third_party/scs_paper/`. The adapter disables progress callbacks, records
separate compilation/warmup clocks, and accumulates radial estimates in bounded
memory while invoking the unmodified author transition. The current 100D
comparison implements DCS in the same JAX scan framework, with equivalent state
recording and projection work. Both maps use our common VI fitting protocol and
both methods receive the same separate, acceptance-based step-size tuning.
This applies the author's SCS transition to the original DCS normal-CDF skew
target; it does not reproduce the author's original tuning recipe, Student-t-CDF
skew target or ν=1/2 figures. See [the comparison protocol](docs/SCS_100D.md).

NumPy, SciPy, JAX, NumPyro, BlackJAX, ArviZ and Matplotlib are installed as dependencies and retain their own licenses. Cite relevant computational libraries when reporting the experiments.
