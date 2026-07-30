# Data provenance

All model specifications and ground-truth/reference quantities used by this
benchmark suite come from PosteriorDB. The original simulation folder did not
record the exact PosteriorDB identifier and revision for every experiment, so
those immutable identifiers must be added before the paper release.

## GARCH(1,1) benchmark

`raw/garch11.json` is the 200-observation data payload used by the GARCH(1,1)
experiment.  It has the keys:

| Key | Meaning |
| --- | --- |
| `T` | Number of observations (`200`) |
| `y` | Observed time series |
| `sigma1` | Initial standard deviation (`0.5`) |

The payload was obtained from PosteriorDB's `garch` data archive:

- metadata: <https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database/data/info/garch.info.json>
- source archive: <https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database/data/data/garch.json.zip>

The URLs above use PosteriorDB's moving `master` branch.  The bundled payload
is the release artifact: its SHA-256 is checked in the repository tests, while
the runtime loader validates the JSON schema and `T == len(y)`.  During this
repository cleanup, its JSON values were compared against the extracted
upstream payload; that one-time comparison is not performed at import.  The
bundled file is formatted for review, so its byte representation differs from
the extracted upstream JSON.  Checksums recorded during preparation are:

```text
Bundled garch11.json:          0690040d698496babaa4e5727b08fb978339fb87d91c4a2d9cfe76ab363872fc
Extracted upstream garch.json: 6f5dd0d191c8f5f0dd655d1936a78dded65c41807d9c733e18d08a2fb0437ce0
```

To use a separately obtained copy, set `HEAVY_TAILS_GARCH_DATA` to a JSON file
with the same three keys.  The loader validates the declared `T` against the
length of `y` before a run starts.  Before a final public archive, also record
the exact upstream PosteriorDB commit or release from which the data were
retrieved.

## PosteriorDB reference summaries

The Eight Schools and GARCH modules contain fixed posterior means and second
moments from PosteriorDB. They are embedded benchmark inputs, not freshly
downloaded values. Before paper release, record the exact PosteriorDB model
and reference-draw revision used to produce them, then verify that their
parameter order and parameterization match the code. See
[`docs/VALIDATION.md`](../docs/VALIDATION.md).
