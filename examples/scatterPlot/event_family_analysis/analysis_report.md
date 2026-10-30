# Event-family discovery report

## Scope and traceability

- Catalogue runs: 13; sampled events: 156,000; usable events: 156,000.
- Sampling is equal-run, deterministic hash stratification—not pooled-catalogue sampling. It therefore does not let permissive configurations dominate descriptor geometry.
- `event_assignments.csv` retains IDs, location, bounds, raw/transformed descriptors, PCA, exclusions and family assignment. `-1` is intentionally unassigned/noise, not discarded.

## Descriptor preparation

- Features: log duration, three threshold/diurnal fractions, log mean positive excess (heat content / duration), and two nocturnal fractions. Dividing cumulative heat content by duration avoids a near-mechanical duration duplicate.
- Median/IQR scaling and clipping to ±6 scaled IQRs were applied before PCA/density geometry. Invalid rows remain in the assignment file with an exclusion reason.
- PCA explained variance: PC1=0.674, PC2=0.183, PC3=0.089, PC4=0.033, PC5=0.021, PC6=0.000.
- Exact-zero fractions: log_duration_hours=0.000, fraction_above_threshold=0.000, daytime_strong_exceedance_fraction=0.013, log_mean_excess_degC=0.000, nocturnal_below_climatology_fraction=0.953, nocturnal_below_threshold_fraction=0.225. The mostly-zero nocturnal-below-climatology descriptor is retained for traceability but has negligible PCA variance.
- Strongest Spearman relationships: fraction_above_threshold/nocturnal_below_threshold_fraction=-0.881; daytime_strong_exceedance_fraction/log_mean_excess_degC=0.832; fraction_above_threshold/log_mean_excess_degC=0.754; fraction_above_threshold/daytime_strong_exceedance_fraction=0.753. These indicate an intensity/persistence axis, not independent evidence for separate families.
- PCA loadings (rows PC1...):

````
[[-0.26  -0.526 -0.367 -0.468  0.001  0.55 ]
 [ 0.813 -0.307  0.172  0.23   0.001  0.401]
 [ 0.515  0.154 -0.35  -0.655 -0.001 -0.4  ]
 [ 0.029 -0.404 -0.658  0.51  -0.002 -0.378]
 [-0.069 -0.665  0.529 -0.198  0.001 -0.483]
 [ 0.     0.001 -0.002  0.001  1.    -0.002]]
````

## Conservative density-family result

- Baseline: k=25, core-radius quantile=0.72, epsilon=0.2203, core fraction=0.720, min component=120; families=1; assigned fraction=0.870.
- This is a dependency-free density graph, not HDBSCAN. Border events cannot link two cores, so apparent families are conservative hypotheses rather than forced classifications.
- Result: the sampled descriptor space supports one broad, connected event population—not multiple separable event families. Its stability means this negative result is more defensible than a visually attractive forced taxonomy.
- Stability checks on the same events:
  - k20_q68: epsilon=0.2028, core=0.680, families=1, assigned=0.837, ARI=0.823, reference-family Jaccard=0.960.
  - k30_q76: epsilon=0.2379, core=0.760, families=1, assigned=0.899, ARI=0.827, reference-family Jaccard=0.968.
  - rank-uniform normalization (k25_q72): epsilon=0.0973, core=0.720, families=4, assigned=0.887, ARI=0.038, reference-family Jaccard=0.603.
- Conclusion: the rank-uniform check produces a materially different partition. Therefore the apparent rank-scale families are not robust to reasonable normalization, and neither partition is currently defensible as a physical event taxonomy.

## Interpretation guardrails

- A separated PCA region is descriptor-space evidence only. Validate every proposed family by geographic footprint and time evolution before asserting a physical mechanism.
- A low stability score, single-run concentration, or descriptor-boundary family is evidence against a persistent physical taxonomy.
- UMAP/HDBSCAN unavailable (not installed); no dependency was added.
