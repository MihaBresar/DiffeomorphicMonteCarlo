# Third-party attribution

The repository's original code is released under the MIT license. This does not relicense third-party material.

**PosteriorDB.** Data, model specifications, metadata and reference posterior draws are from [stan-dev/posteriordb](https://github.com/stan-dev/posteriordb/tree/5545a1dd07ae297c36edecbcd82aa49097b4c385), revision `5545a1dd07ae297c36edecbcd82aa49097b4c385`. Its BSD 3-Clause copyright and terms are preserved in `data/posteriordb/LICENCE.md`. The bundled model-info records identify the models as BSD3. Cite the PosteriorDB paper and the original model sources identified in the metadata.

**Sub-Cauchy Sampler (SCS).** The active 100D/500D comparisons call the
unmodified author projection, variational optimizer and step-out transition from
[liusf15/sub-cauchy-sampler](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565),
pinned at paper-date revision `ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`.
Byte-for-byte source copies, SHA-256 checksums, upstream Git blob identities and
Sifan Liu's MIT copyright notice are preserved in `third_party/scs_paper/`.
The [SCS paper](https://arxiv.org/abs/2601.11066) is by Sebastiano Grazzi,
Sifan Liu, Gareth O. Roberts and Jun Yang.

The adapter removes progress callbacks, compiles before timing, and evaluates
radial summaries in bounded scan blocks. The active version-11 baseline uses
the authors' VI recipe and fixed h=0.1; an explicit acceptance-tuned alternative
is retained. Full settings use the authors' 1,000 VI updates on 2,000 fixed
reference draws. Smaller presets reduce that fitting budget. Version 10 gives DCS the same 1,000-update/2,000-point full budget and a
fixed reference cloud of its own. Its parameterization, gradient clipping
and calibration recipe remain distinct. See [the complete protocol](docs/COMPARISONS.md)
and [the author-source attribution](third_party/scs_paper/README.md).

All methods evaluate the same benchmark target. The normal-CDF-skewed t(3)
target here differs from the authors' Student-t-CDF skew targets at ν=1 and ν=2;
these comparisons do not reproduce the SCS paper's figures. JAX is the common
framework comparison. Optimized Numba DCS versus author JAX SCS is separately
labelled as an implementation comparison.

Every VI-enabled SCS map fit calls the pinned author optimizer, including
logistic and the ν=0.5 experiment. The generic experiments retain independently
checked NumPy/Numba SCS production implementations, derived from the supplied experiments and inspected against
upstream revision `25ea5ee21ae93a1737bbc34bd18a763f191f85de`. They are not
represented as unmodified author performance implementations. Their
acceptance-target 0.234 step calibration is unchanged. Existing saved results
retain the source revisions and fitting recipes that generated them.

NumPy, SciPy, JAX, NumPyro, BlackJAX, ArviZ and Matplotlib are installed as dependencies and retain their own licenses. Cite relevant computational libraries when reporting the experiments.
