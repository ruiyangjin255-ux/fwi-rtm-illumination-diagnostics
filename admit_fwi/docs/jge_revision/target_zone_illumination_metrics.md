# Target-zone illumination and FWI-RTM diagnostics

The zones are derived from the SEG/Salt high-velocity body: salt top, salt flanks, and subsalt shadow. Metrics connect illumination, RTM image response, and FWI update energy.

| Zone | Pixels | Src illum | Rec illum | Src-rec illum | Low illum frac | Src-norm RTM | Lap RTM | Full update | Damped update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| salt_top | 5168 | 0.7223 | 0.5322 | 0.6174 | 0.0000 | 0.5456 | 0.1744 | 0.185 | 0.018 |
| salt_flanks | 6389 | 0.5952 | 0.3160 | 0.4284 | 0.0000 | 0.3907 | 0.1955 | 0.003 | 0.000 |
| subsalt_shadow | 16150 | 0.3689 | 0.1102 | 0.2011 | 0.0000 | 0.1811 | 0.0880 | 0.000 | 0.000 |

## Interpretation

- The table makes the paper framework operational: RTM illumination, migrated-image response, and FWI update energy are evaluated over the same target zones.
- The damped update should not be read as high-quality velocity recovery; it is a controlled update passed through quality gates before RTM validation.
- Subsalt and flank metrics are the most relevant zones for illumination compensation claims.
