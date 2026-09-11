# Data and reference provenance

The synthetic skew-t targets and separable logistic dataset are generated locally from explicit formulas/seeds; they do not come from PosteriorDB.

Eight Schools and GARCH use the snapshot of [stan-dev/posteriordb](https://github.com/stan-dev/posteriordb/tree/5545a1dd07ae297c36edecbcd82aa49097b4c385) at revision `5545a1dd07ae297c36edecbcd82aa49097b4c385`. The bundled `posteriordb/manifest.json` records exact URLs and SHA-256 checksums for data, Stan models, posterior records, model metadata, reference metadata and reference draws. Archives are read without extracting their paths. Data/reference checksums are verified on loading. `python scripts/fetch_posteriordb.py` checks the bundle and restores missing files from the pinned revision.

- Eight Schools posterior: `eight_schools-eight_schools_centered`; reference record `eight_schools-eight_schools_noncentered`, transformed physical parameters θ₁,…,θ₈,μ,τ.
- GARCH posterior/reference: `garch-garch11`; physical order μ,α₀,α₁,β₁. This is simulated time-series data, as the bundled metadata states.
- Both reference archives contain 10 chains of 1,000 retained draws. These are Monte Carlo references with finite uncertainty; the summary estimates functional MCSE from the saved chains.

The models specify BSD3 licensing. The upstream BSD license is retained in `posteriordb/LICENCE.md`; see [THIRD_PARTY.md](../THIRD_PARTY.md). The old `raw/garch11.json` is a legacy data copy retained for provenance and is not used by the corrected engine.
