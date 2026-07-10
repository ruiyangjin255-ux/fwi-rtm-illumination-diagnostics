# Illumination-normalized acoustic reverse-time migration for improved SEG/Salt model imaging

## Abstract

Reverse-time migration (RTM) is well suited for imaging complex structures with strong lateral velocity contrasts, but its practical image quality is often controlled not only by wave-equation accuracy but also by illumination imbalance and low-wavenumber artifacts. We present a reproducible two-dimensional acoustic RTM workflow that combines high-order finite-difference wave propagation, prestack multi-shot zero-lag cross-correlation imaging, source and source-receiver illumination normalization, and Laplacian-based post-imaging enhancement. The workflow was evaluated on a 2-D SEG/Salt velocity model with 676 horizontal samples and 230 depth samples, using a 10 m grid spacing, a 1 ms time step, 4001 time samples, a 20 Hz Ricker source, and 224 shots. A display-diagnostic analysis showed that low-illumination zones occupied only 1.74% of the model grid, indicating that the full-aperture result was primarily limited by display compression and low-frequency image components rather than by severe illumination loss. In the full scheme-2 run, source-receiver illumination normalization remained highly correlated with source-only normalization (correlation coefficient 0.9732), whereas Laplacian source-normalized imaging reduced the low-wavenumber background and clarified salt-boundary and sedimentary reflector geometry. These results indicate that, for the tested SEG/Salt configuration, conservative display balancing provides the most stable main migration image, while Laplacian-enhanced illumination-normalized imaging is useful as a complementary structural interpretation panel. The implementation also includes checkpoint-based multi-shot accumulation, enabling interrupted large RTM runs to resume without recomputing completed shots.

**Keywords:** reverse-time migration; acoustic wave equation; finite difference; illumination compensation; Laplacian filtering; SEG/Salt model; seismic imaging

## 1. Introduction

Seismic migration aims to reposition recorded wavefield energy to its subsurface reflection locations. Among migration methods, reverse-time migration is attractive because it uses two-way wave-equation propagation and can therefore handle steep dips, overturned waves, and strong velocity contrasts more naturally than one-way extrapolation or ray-based migration. Since the early development of RTM, this property has made it especially relevant for salt-related structures, where sharp velocity contrasts and complex wavepaths often degrade conventional imaging.

Despite this advantage, RTM images are not automatically interpretation-ready. A standard prestack RTM image is usually formed by correlating a forward-propagated source wavefield with a backward-propagated receiver wavefield at zero time lag. This zero-lag cross-correlation imaging condition is robust and easy to implement, but it is sensitive to uneven source and receiver illumination. In addition, the cross-correlation image can contain strong low-wavenumber components, backscattering noise, and depth-dependent amplitude variations. These effects can obscure salt flanks, subsalt reflectors, and weak sedimentary layers even when the wave propagation itself is correct.

Illumination compensation and image-domain filtering are common practical strategies for improving RTM readability. Source illumination normalization reduces amplitude bias caused by uneven source wavefield energy. Source-receiver normalization further accounts for the receiver wavefield distribution, but it may amplify poorly illuminated zones if not controlled. Laplacian filtering can suppress low-wavenumber artifacts and sharpen reflector boundaries, but it may also over-emphasize local edges if used as the only final image. Therefore, a useful RTM workflow should distinguish between physical imaging-condition limitations and display/post-processing limitations.

This study develops and evaluates a reproducible 2-D acoustic RTM workflow for the SEG/Salt model. The contribution is not a new wave-equation solver, but an integrated and testable imaging pipeline that compares conservative display optimization with physically motivated source-receiver illumination normalization and Laplacian-enhanced imaging. We use the same full multi-shot acquisition geometry to assess whether the salt-model result is limited mainly by illumination or by display and image-conditioning choices. The evidence shows that full-aperture illumination is sufficient for most of the model, and that Laplacian-enhanced normalized imaging improves structural readability while requiring careful interpretation.

## 2. Method

### 2.1 Workflow overview

The workflow contains five modules: velocity-model preparation, acoustic finite-difference modeling, prestack reverse-time migration, illumination-normalized imaging-condition analysis, and paper-style display optimization. The input is a 2-D acoustic velocity model. The output is a set of migration images: the raw cross-correlation image, source-normalized image, source-receiver-normalized image, Laplacian image, Laplacian source-normalized image, and conservative display-optimized image.

The main processing sequence is:

1. Read the SEG/Salt velocity model in binary x-major format and convert it to a depth-by-distance array.
2. Pad the velocity model laterally and at the bottom to reduce boundary effects.
3. Smooth the velocity model to form the migration/background velocity.
4. For each shot, forward-model the source wavefield and surface receiver record.
5. Optionally subtract the direct wavefield generated in the smoothed migration velocity.
6. Reverse-propagate the receiver wavefield and apply zero-lag source-receiver cross-correlation.
7. Accumulate the shot images, source illumination, receiver illumination, and stacked records.
8. Generate normalized and Laplacian-enhanced image candidates.
9. Apply conservative display balancing for publication-style figures.

### 2.2 Acoustic finite-difference modeling

Wave propagation is modeled with a 2-D acoustic finite-difference scheme. The implementation uses an explicit time-stepping formulation with high-order spatial finite-difference coefficients for the Laplacian operator. A Ricker wavelet is injected at the source position, and a surface receiver line records pressure wavefields at the receiver depth. Absorbing boundary masks are applied around the computational domain, with the option to leave the top boundary unabsorbed when needed.

For the SEG/Salt experiment, the model contains 676 horizontal grid points and 230 depth grid points. The grid spacing is 10 m in both directions. The time step is 0.001 s, and the full run uses 4001 time samples. The source peak frequency is 20 Hz. The migration run uses lateral padding of 60 grid cells and bottom padding of 60 grid cells, giving a padded computational grid of 796 by 290 samples. The finite-difference order is 8, and the absorbing boundary width is 40 cells.

### 2.3 Prestack RTM imaging condition

For each shot, the source wavefield is propagated forward in time and saved to disk. The receiver wavefield is then propagated backward in time from the recorded surface data. The migrated image is formed by zero-lag cross-correlation:

\[
I(\mathbf{x}) = \sum_t S(\mathbf{x},t)R(\mathbf{x},t),
\]

where \(S(\mathbf{x},t)\) is the source wavefield and \(R(\mathbf{x},t)\) is the receiver wavefield at location \(\mathbf{x}\) and time \(t\). For multi-shot migration, shot images are stacked:

\[
I_{\mathrm{stack}}(\mathbf{x}) = \sum_s I_s(\mathbf{x}).
\]

The workflow also accumulates source illumination,

\[
L_s(\mathbf{x}) = \sum_{s,t} S_s^2(\mathbf{x},t),
\]

and receiver illumination,

\[
L_r(\mathbf{x}) = \sum_{s,t} R_s^2(\mathbf{x},t).
\]

These illumination fields are used to evaluate whether the final image is limited by acquisition/wavefield coverage or mainly by display and post-imaging processing.

### 2.4 Illumination normalization and Laplacian enhancement

The source-normalized image is computed as:

\[
I_s^{N}(\mathbf{x}) =
\frac{I_{\mathrm{stack}}(\mathbf{x})}{L_s(\mathbf{x})+\epsilon},
\]

where \(\epsilon\) is a small stabilization term. To include receiver-side energy distribution, the source-receiver-normalized image is computed as:

\[
I_{sr}^{N}(\mathbf{x}) =
\frac{I_{\mathrm{stack}}(\mathbf{x})}
{\sqrt{L_s(\mathbf{x})L_r(\mathbf{x})}+\epsilon}.
\]

A minimum illumination fraction is used to suppress unstable normalization in very poorly illuminated cells. In the present full-aperture SEG/Salt run, the low-illumination fraction below 1% of the maximum geometric illumination is 0.0174, indicating that most of the model is sufficiently illuminated.

To reduce low-wavenumber RTM artifacts, the workflow also applies a Laplacian operator to the stacked image:

\[
I_{\Delta}(\mathbf{x}) = \nabla^2 I_{\mathrm{stack}}(\mathbf{x}),
\]

and then applies source normalization to produce a Laplacian source-normalized image. This candidate image is intended for structural interpretation and comparison rather than as an unqualified replacement for the conservative migration display.

### 2.5 Checkpoint-based multi-shot accumulation

Full multi-shot RTM is computationally expensive. To make the run recoverable, the implementation writes checkpoint files after completed shots. The checkpoint stores cumulative image, source illumination, receiver illumination, stacked record, and a manifest containing the model and acquisition signature. When a run is resumed, completed padded shot positions are skipped and the remaining shots are computed. This design prevents recomputation after interruption and was used in the full scheme-2 run.

## 3. Experimental setup

### 3.1 Model and acquisition geometry

Experiments were conducted on a 2-D SEG/Salt acoustic velocity model. The physical grid size was 676 by 230 samples with 10 m spacing. The full scheme-2 experiment used 224 shot positions with approximately 30 m shot spacing. Source and receiver depths were both four grid cells below the surface. The time step was 0.001 s, the number of time samples was 4001, and the source wavelet peak frequency was 20 Hz.

The migration velocity was generated by smoothing the padded velocity model with two passes of a separable box filter using 10-cell horizontal and vertical radii. Direct-wave subtraction and direct-arrival muting were used to reduce near-surface and direct-arrival dominance before receiver-wavefield back propagation.

### 3.2 Compared imaging outputs

Two groups of outputs were compared. Scheme 1 used the existing full RTM result and optimized only display and paper-style post-processing. It did not change the acoustic propagator, the zero-lag cross-correlation imaging condition, or the physical wavefield simulation. Scheme 2 recomputed the full RTM image with additional receiver illumination accumulation, source-receiver illumination normalization, and Laplacian image candidates.

The primary scheme-1 output is the conservative paper-ready migration image. The primary scheme-2 outputs are the source-normalized image, source-receiver-normalized image, Laplacian image, and Laplacian source-normalized image. The comparison is therefore designed to separate display improvement from imaging-condition improvement.

## 4. Results

### 4.1 Display diagnostics indicate that the full image is not dominated by illumination loss

The initial diagnostic analysis classified the existing full migration result as display-dominated. The low-illumination fraction was 0.0174, while the lateral normalized energy values were 21.3791, 21.6368, and 26.1118 for the left, middle, and right parts of the image, respectively. Depth-zone normalized energies were 25.1333, 21.3961, and 22.585 for shallow, middle, and deep zones. These values indicate lateral and depth-dependent amplitude variation, but not a severe loss of illumination over most of the model.

The scheme-1 paper-ready image therefore used conservative symmetric clipping and display balancing. This display strategy improved readability while avoiding aggressive depth-dependent gains that could create artificial continuity. The result provides a stable main migration panel for publication-style presentation.

### 4.2 Source-receiver illumination normalization preserves the main image structure

In the full scheme-2 run, the source-receiver-normalized image was highly correlated with the source-normalized image, with a correlation coefficient of 0.9732. This high correlation shows that receiver-side illumination compensation does not fundamentally change the main migrated structure under the tested full-aperture geometry. The geometric low-illumination fraction remained 0.0174, consistent with the scheme-1 diagnostic.

Visually, source-receiver normalization slightly improves lateral amplitude balance but does not reveal a substantially different salt geometry. The salt top, salt flanks, and surrounding sedimentary reflectors remain consistent with the source-normalized image. Therefore, source-receiver normalization should be interpreted as a physically motivated amplitude-balancing variant rather than a decisive new image in this full-data setting.

### 4.3 Laplacian source-normalized imaging improves structural readability

The Laplacian source-normalized image shows clearer reflector edges and salt-boundary features than the conservative source-normalized image. The correlation between the Laplacian source-normalized image and the source-normalized image is -0.2822, which reflects the different spectral and phase characteristics introduced by the Laplacian operator. Low-wavenumber background components are suppressed, while sedimentary layers, salt flanks, and discontinuities become more visually prominent.

This improvement is most useful for structural interpretation. However, because Laplacian filtering is an image-domain enhancement, it should not be used as the only final migrated image. The conservative scheme-1 display is more stable for main-result presentation, whereas the Laplacian source-normalized image is better suited as a complementary panel that highlights boundaries and local structural details.

### 4.4 Checkpointing enables recoverable full RTM computation

The full scheme-2 run was completed with 224 shots and four workers. During the computation, cumulative checkpoint arrays and a manifest were written after completed shots. After interruption, the workflow resumed from the saved checkpoint and skipped previously completed shots. This checkpoint mechanism does not alter the imaging condition, but it improves computational reproducibility and robustness for long RTM experiments.

## 5. Discussion

The comparison between scheme 1 and scheme 2 suggests that the main limitation of the tested SEG/Salt result is not insufficient illumination but the presentation and conditioning of the RTM image. The full geometry produces adequate source-receiver coverage over most of the model, as indicated by the low-illumination fraction of 1.74%. Consequently, source-receiver normalization yields only moderate amplitude-balancing changes relative to source-only normalization.

The most useful scheme-2 contribution is the Laplacian source-normalized image. By suppressing low-wavenumber background energy, it makes salt boundaries and stratigraphic reflectors easier to trace. This is valuable for interpretation, especially where the conservative display makes weak reflectors difficult to see. Nevertheless, Laplacian enhancement can also increase the apparent sharpness of local events, so it should be reported as an enhanced interpretation image rather than as a replacement for the primary migrated section.

The present study has several limitations. First, the evaluation is based on a single 2-D SEG/Salt model and does not yet include multiple benchmark models such as Marmousi, Overthrust, or field data. Second, the comparison is primarily image-domain and qualitative, with limited quantitative metrics beyond illumination fraction, correlation, and energy balance. Third, the workflow uses acoustic propagation and therefore does not account for elastic mode conversion, anisotropy, attenuation, or density variations. These limitations define the scope of the current implementation and motivate future extensions.

Future work should include quantitative comparison against additional migration variants, such as deconvolution imaging conditions, least-squares RTM, or angle-domain gathers. Additional evaluation metrics could include structural similarity to the known velocity interfaces, reflector continuity measures, image sharpness, and noise-level estimates. Extending the workflow to elastic or anisotropic modeling would also make the approach more relevant to field-scale salt imaging.

## 6. Conclusion

This work presents a reproducible 2-D acoustic RTM workflow for SEG/Salt imaging that integrates high-order finite-difference propagation, multi-shot zero-lag cross-correlation imaging, source and source-receiver illumination normalization, Laplacian enhancement, and checkpoint-based computation. The full 224-shot experiment shows that source-receiver normalization preserves the main migrated structure and that illumination loss is not the dominant limitation under the tested acquisition geometry. Laplacian source-normalized imaging provides clearer salt-boundary and reflector detail, while conservative display optimization remains the most stable choice for the main publication image. The resulting workflow provides both an interpretable set of image candidates and a recoverable computational framework for larger RTM experiments.

## Figure captions

**Figure 1. SEG/Salt acoustic RTM workflow.** The workflow reads the binary velocity model, pads and smooths the migration velocity, forward-propagates source wavefields, reverse-propagates receiver wavefields, applies zero-lag cross-correlation, accumulates multi-shot images and illumination fields, and generates source-normalized, source-receiver-normalized, Laplacian, and paper-ready display outputs.

**Figure 2. Conservative scheme-1 paper-ready RTM image.** The image is produced from the existing full RTM result using display-only optimization. The processing uses conservative symmetric clipping and display balancing to improve readability without changing the underlying wave-equation migration result.

**Figure 3. Scheme-2 imaging-condition comparison.** The panels show source-normalized imaging, source-receiver-normalized imaging, Laplacian source-normalized imaging, and receiver illumination. Source-receiver normalization preserves the main structure, whereas Laplacian source-normalized imaging enhances reflector and salt-boundary details.

**Figure 4. Checkpoint-based full multi-shot computation.** The checkpoint mechanism stores cumulative image, source illumination, receiver illumination, stacked record, and a manifest of completed shots. Interrupted full RTM runs can resume by skipping completed shot positions.

## Tables

**Table 1. Full SEG/Salt RTM parameters.**

| Parameter | Value |
|---|---:|
| Grid size | 676 x 230 |
| Grid spacing | 10 m x 10 m |
| Time step | 0.001 s |
| Number of time samples | 4001 |
| Source peak frequency | 20 Hz |
| Number of shots | 224 |
| Shot spacing | approximately 30 m |
| Source depth | 4 grid cells |
| Receiver depth | 4 grid cells |
| Padding | 60 cells laterally and 60 cells at the bottom |
| Finite-difference order | 8 |
| Workers | 4 |

**Table 2. Main diagnostic metrics.**

| Metric | Scheme 1 / diagnostic | Scheme 2 full |
|---|---:|---:|
| Low-illumination fraction | 0.0174 | 0.0174 |
| Source-receiver vs source-normalized correlation | not applicable | 0.9732 |
| Laplacian source-normalized vs source-normalized correlation | not applicable | -0.2822 |
| Receiver illumination maximum | not applicable | 61,900,264 |

## References

1. Baysal, E., Kosloff, D. D. & Sherwood, J. W. C. Reverse time migration. *Geophysics* **48**, 1514-1524 (1983). DOI: 10.1190/1.1441434.
2. Claerbout, J. F. Toward a unified theory of reflector mapping. *Geophysics* **36**, 467-481 (1971).
3. Rickett, J. E. The imaging condition. Stanford Exploration Project Report 115 (2002).
4. Liu, F. et al. Imaging conditions for prestack reverse-time migration. *SEG Technical Program Expanded Abstracts* (2007).
5. SEG/EAGE Salt model studies and acoustic RTM benchmark applications.

## 中文写作说明

这篇初稿的主张应保持克制：本文不是提出全新的 RTM 理论，而是提出一个可复现、可断点续跑的二维声波 RTM 工作流，并比较显示优化、源-检照明归一化和 Laplacian 增强对 SEG/Salt 成像结果的影响。当前证据支持的核心结论是：full 224 炮情况下照明不是主控问题，方案一适合作为稳定主图，方案二的 Laplacian 图适合作为结构细节增强图。

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Full-aperture result is not dominated by illumination loss. | Low-illumination fraction is 0.0174 in both diagnostics and scheme-2 full run. | Supported |
| Source-receiver normalization does not substantially alter the main image structure. | Correlation with source-normalized image is 0.9732. | Supported |
| Laplacian source-normalized imaging improves boundary readability. | Visual comparison shows sharper salt boundary and reflector detail; correlation with source-normalized image is -0.2822, indicating a distinct high-wavenumber image character. | Supported qualitatively; needs stronger quantitative metric |
| Checkpointing improves computational robustness. | Full run resumed from saved checkpoint and completed 224 shots. | Supported by implementation and run log |
| The method generalizes to other models. | No additional model tested yet. | Needs evidence |

