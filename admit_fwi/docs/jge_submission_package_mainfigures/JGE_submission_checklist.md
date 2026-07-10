# JGE Submission Checklist

## Ready items

- Manuscript draft is included in Markdown format.
- Figure captions are included separately.
- Figure alt text is included for final OUP/JGE template conversion.
- Core result tables are included as CSV files.
- Innovation claims are mapped to programs, evidence files, figures, and claim boundaries.
- Figure 1-5 files are packaged in the requested formats.
- Optimized FWI-RTM pipeline report is included.

## Key result gate

- `selected_alpha`: 0.1
- `rtm_verdict`: after_fwi_closer_to_reference
- `filtered_rmse_before`: 0.02713028664829544
- `filtered_rmse_after`: 0.027109411257509722
- `filtered_rmse_improvement_fraction`: 0.0007694496949604249

## Manual checks before journal submission

- Convert the manuscript to the target JGE/OUP Word or LaTeX template.
- Recheck JGE word limit, abstract length, keywords, figure count, and reference style against the current author guidelines.
- Verify every DOI, page number, author spelling, and in-text citation.
- Confirm that SEG/Salt model redistribution is allowed before uploading any raw model data.
- Keep claims conservative: the result supports a diagnostic quality-gated FWI-RTM workflow, not production-grade FWI velocity recovery.
- If using TIFF figures, confirm final journal-required DPI after layout scaling.
