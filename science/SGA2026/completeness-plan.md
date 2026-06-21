# SGA-2026/27 Completeness Plan: Extending to Smaller Angular Diameters

**Status:** Working draft (2026-06-21)  
**Author:** John Moustakas  
**Collaborators:** Viraj Manwadkar (Stanford/SLAC), Risa Wechsler (Stanford)

---

## 1. Motivation

SGA-2025 took nearly three years to complete, bottlenecked almost entirely by the visual inspection (VI) of candidate large galaxies drawn from external extragalactic catalogs (NED, HyperLeda, NED-LVS, etc.). The resulting sample is diameter-limited at roughly D26 > 25 arcsec (varying by region), leaving a significant completeness gap at smaller angular diameters, particularly in the 15–45 arcsec range. This size regime contains a large, scientifically important population: star-forming dwarf irregulars, compact spheroidals in clusters, and low-luminosity late-type spirals that are genuinely nearby but small on the sky.

Three developments now make it practical to push to smaller diameters without repeating the VI bottleneck:

1. **DR11 Tractor catalogs are complete** over the full Legacy Survey footprint. The morphological parameters (`FRACFLUX`, `RCHISQ`, `TYPE`, `SHAPE_R`, etc.) provide quantitative shredding and morphology diagnostics without any spectroscopic requirement.
2. **SSL embeddings** (MoCo v2 contrastive, 152×152 px grz) exist for the full SGA-2025 sample and for Manwadkar's DESI DR9 dwarf galaxy catalog. These provide a redshift-independent appearance-based similarity metric.
3. **Manwadkar et al. (in prep.)** — "When Galaxies Fall Apart: Addressing Photometric Shredding with Color-Based Aperture Photometry in DESI DR1" — establishes a validated pixel-level shredding reconstruction pipeline. Code: https://github.com/virajvman/desi_dwarfs

The goal is a pipeline that identifies, reconstructs, and photometers the new candidates with targeted VI only on the uncertain tail — not on every candidate.

---

## 2. The Two Candidate Populations

The 15–45 arcsec completeness problem has two distinct flavors that require different entry points.

### 2A. Shredded Galaxies (FRACFLUX pathway)

Blue, irregular, star-forming dwarfs are disproportionately shredded by Tractor because they violate the single-peak, monotonically-decreasing surface brightness assumption. A galaxy with `FRACFLUX > 0.2` in 2+ bands (grz) is a likely shred: its photometry is split across multiple Tractor sources, each of which individually falls below the SGA angular-diameter threshold even though the parent galaxy would not.

Key statistics from Manwadkar et al.: at z < 0.01, 25–70% of BGS/ELG sources are likely shreds; ~80% of galaxies with M* ~ 10^6 Msun are shreds in BGS Bright. The problem is severe and well-characterized.

### 2B. Isolated Compact Spheroidals (Tractor morphology pathway)

Bright, compact early-type galaxies — dwarf ellipticals, compact lenticulars, infall spheroidals in clusters — may have clean, single-component Tractor fits (`FRACFLUX ≤ 0.2`, `TYPE = DEV` or `SER`) but fall below the SGA-2025 diameter threshold. These are the "easy" cases photometrically but require their own selection logic: they are not shredded, just small. Clusters (Virgo, Fornax, Coma, Perseus, etc.) are particularly important hunting grounds.

---

## 3. Stage 0: Pre-filtering (DR11 Tractor)

Starting point: all DR11 Tractor sources with `TYPE ≠ PSF` (extended sources only).

**Cuts to apply before any ML step:**

- **Bright star masking**: apply Legacy Survey bright star masks (Gaia + Tycho-2 based). The radius–magnitude relation used by Tractor (Manwadkar eq. 3) is a good starting point. Sources within these masks are excluded. This is the single most important false-positive reducer: >90% of spurious "large" Tractor sources near bright stars are removed here.
- **Galactic cirrus rejection**: SFD E(B-V) threshold (to be tuned; start with E(B-V) < 0.15 mag) plus Legacy Survey quality bitmasks that flag cirrus-affected regions. Aggressive galactic latitude cut (|b| > 20°, tbd) as a secondary filter.
- **Angular size pre-selection**: for the FRACFLUX pathway, no explicit size cut yet (shreds may have small individual `SHAPE_R`). For the spheroidal pathway, select sources with `SHAPE_R` corresponding to r_eff > 3–4 arcsec (roughly D26 > 10–12 arcsec) as a lower bound.
- **Color sanity**: exclude sources with colors inconsistent with low-redshift galaxies (e.g., very blue point-like sources, very red stellar locus objects). At low redshift, (g-r)_rest ≈ (g-r)_obs (dust-corrected), so a simple observed-frame color cut is sufficient.

Expected reduction from Stage 0: ~10–50× compression of the full DR11 extended source catalog, before any ML.

---

## 4. Stage 1A: Shredded Galaxy Reconstruction (FRACFLUX Pathway)

**Entry criterion**: `FRACFLUX > 0.2` in 2+ grz bands AND `RCHISQ < 4` (to exclude genuinely bad fits that are not shredding).

Re-implement Manwadkar's pipeline within the SGA infrastructure (`py/SGA/shredding.py` or similar). The core steps:

1. **Image cutout**: centered on the Tractor source; size set by expected angular diameter at the estimated redshift (use photo-z from the Legacy Survey if available; otherwise use a fixed conservative size, e.g., 1.5 arcmin, matching Manwadkar's z > 0.0125 default).

2. **Image segmentation** (photutils): combine grz image; threshold = 1.5σ_bkg, npixels = 10. Identify the "main blob" containing the Tractor source.

3. **Bright star + bad pixel masking**: Gaia-based star masks (see Stage 0); flag zero-ivar pixels and dilate mask by 4 pixels.

4. **Color-based association**: at low redshift, (g-r)_rest ≈ (g-r)_obs (dust-corrected), so the full galaxy population color distribution at the source redshift can be approximated from observed colors without a K-correction. Associate deblended blobs and Tractor sources with the parent galaxy using:
   - (g-r)_i ≤ (g-r)_ref + 0.1 and (r-z)_i ≤ (r-z)_ref + 0.1 (bluer sub-regions always included)
   - Failing that: within 2.5σ of the galaxy color distribution at the estimated redshift
   - Fallback (no photo-z available): `SIMPLE` photometry — sum all Tractor sources on the main blob without any color cut

5. **Background/blend subtraction**: subtract Tractor models of non-associated sources; mask 5σ residuals outside the main blob.

6. **Isolation mask** (prevent over-merging): sweep ncontrast = 0.05–0.20, track Jaccard score of deblended components, stop when components are stable. Mask distinct neighbor galaxies.

7. **Photometry**: curve-of-growth (CoG) analysis using same empirical model as SGA-2025 (`py/SGA/ellipse.py`). Fallback: `TRACTOR_BASED` (sum associated Tractor fluxes) or `SIMPLE`.

8. **Shapes**: half-light radius and b/a from CoG fit.

**Key reference**: Manwadkar et al. (in prep.). Coordinate with Viraj on code reuse and test cases.

---

## 5. Stage 1B: Compact Spheroidal / Isolated Galaxy Selection (Tractor Morphology Pathway)

**Entry criterion**: `FRACFLUX ≤ 0.2` (not shredded), `TYPE = DEV`, `EXP`, or `SER`, `SHAPE_R` in the target angular size range.

These sources are already well-modeled by Tractor. The challenge is distinguishing nearby low-z galaxies from background E/S0s at higher redshift that happen to fall in the same angular size range.

**Discrimination strategy:**
- **Photo-z cut**: if Legacy Survey photo-z available, require z_phot < 0.1 (most SGA targets)
- **Surface brightness**: nearby dwarfs tend to be high surface brightness for their size; background dwarfs are fainter. Apply a surface brightness vs. apparent size diagnostic.
- **Color cuts**: observed g-r / r-z consistent with low-z galaxy locus (red sequence or blue cloud at z < 0.05).
- **Cluster membership**: cross-match against known cluster catalogs (Virgo, Fornax, Coma, Perseus, etc.). Any extended source near a known SGA group or cluster with consistent colors is a high-priority candidate.

Tractor photometry is already reliable for these sources; the main output of Stage 1B is a ranked candidate list for SSL scoring.

---

## 6. Stage 2: SSL Embedding Similarity Filter

Apply to all surviving candidates from Stages 1A and 1B.

**Anchor embedding set:**
- Full SGA-2025 sample: rescaled 152×152 px grz cutouts (all angular diameters), run through ssl-legacysurvey backbone. Provides morphological diversity across the full size range.
- Manwadkar's DESI DR9 dwarf galaxy catalog: cutouts and pre-computed embeddings (coordinate with Viraj). Extends coverage to the blue, irregular regime most affected by shredding.
- Combined anchor set loaded into a Faiss index (SGAML: `similarity_search_nxn.py`).

**Inference:**
- For Stage 1A candidates: run ssl-legacysurvey inference on the **reconstructed parent galaxy cutout** (post-segmentation), not the raw Tractor detection.
- For Stage 1B candidates: run inference on the individual source cutout, rescaled to standard pixel scale.
- DR11 imaging is DR9-like by construction; the existing ssl-legacysurvey backbone transfers without retraining.

**Scoring:**
- Faiss kNN (k = 50) against the combined anchor set.
- Score = mean cosine similarity to top-k neighbors (or fraction of top-k that are SGA-confirmed galaxies).
- Optionally train an `SSLFineTuner` binary classifier (SGA-confirmed vs. confirmed non-galaxy negatives) for a calibrated probability output.

---

## 7. Stage 3: ZooBot Morphology Filter

Apply ZooBot (pre-trained GZ DECaLS weights, already in SGAML) as an independent second signal.

**Relevant outputs:**
- `smooth-or-featured_smooth`: high value → spheroidal / compact ETG (Stage 1B population)
- `smooth-or-featured_featured-or-disk`: high value → disk/spiral/irregular (Stage 1A population)
- `artifact`: high value → flag for rejection
- Galaxy vs. star/artifact probability (top-level)

No retraining needed initially. Use ZooBot to gate out clear non-galaxies (high artifact probability) and stars that slipped through the Tractor TYPE filter. For ambiguous cases, ZooBot score enters the combined ranking.

Note: ZooBot has not been tested on this application yet. Begin with a small validation sample (known SGA galaxies + known non-galaxies in the 15–45 arcsec range) before deploying at scale.

---

## 8. Stage 4: Combined Scoring and Candidate Ranking

Combine Stage 2 (SSL similarity) and Stage 3 (ZooBot galaxy probability) into a single ranking score. Exact weighting to be calibrated on a validation set.

**Three-tier classification:**

| Tier | Criteria | Action |
|------|----------|--------|
| High-confidence galaxy | SSL score > θ_high AND ZooBot galaxy prob > φ_high | Auto-include in SGA candidate list |
| Uncertain | Neither high-confidence nor low-confidence | Targeted VI |
| Low-confidence (likely non-galaxy) | SSL score < θ_low AND ZooBot galaxy prob < φ_low | Auto-reject |

Thresholds (θ, φ) to be calibrated to achieve: false negative rate < 5% on known SGA galaxies in the size range, false positive rate < 20% overall (i.e., VI burden on the uncertain tier is manageable).

Active learning: confirmed VI results feed back into the SSL anchor set and the ZooBot validation set, shrinking the uncertain tier with each iteration.

---

## 9. Photometry and SGA Integration

**Shredded galaxies (Stage 1A confirmed):**
- CoG photometry from the Manwadkar pipeline (Section 4 above)
- Final validation: compare against Scarlet models for the nearest / most complex cases (as in Manwadkar et al. Section 3.4)
- Integrate into SGA ellipse catalog format (`py/SGA/ellipse.py`)

**Compact spheroidals (Stage 1B confirmed):**
- Tractor photometry is already reliable; adopt directly
- Run SGA forced-photometry (Tractor special mode with fixed ellipse) as for SGA-2025 if diameter warrants it

**New SGA entries:** feed confirmed new galaxies through the full SGA-2025 pipeline (ellipse fitting, multiwavelength photometry, HTML QA page generation) using the existing `SGA2025-mpi` infrastructure.

---

## 10. Known Failure Modes and Mitigations

| Failure mode | Severity | Mitigation |
|---|---|---|
| Highly irregular / LSB galaxies with disconnected blobs | High | SSL embedding on full cutout (not just main blob); Scarlet for validation; flag in catalog |
| Over-merging of distinct galaxies at small separation | Moderate | Isolation mask (Jaccard deblending, Section 4 step 6) |
| Cirrus contamination surviving Stage 0 | Moderate | SSL embeddings distinguish cirrus from galaxies in appearance space; ZooBot artifact flag |
| Bright star halos | Low–Moderate | Stage 0 masks remove most; SSL gating catches residuals |
| Background E/S0s mis-selected as nearby dwarfs (Stage 1B) | Moderate | Photo-z cut; surface brightness diagnostic; SSL similarity to low-z anchor set |
| ZooBot systematic bias toward GZ DECaLS training distribution | Unknown | Validate on known SGA + non-SGA objects before deployment |

---

## 11. Infrastructure and Code

| Component | Location / Tool |
|---|---|
| Shredding reconstruction | `py/SGA/shredding.py` (to be written; based on https://github.com/virajvman/desi_dwarfs) |
| SSL inference | ssl-legacysurvey `predict.py` on SGAML at NERSC |
| SSL similarity search | `similarity_search_nxn.py` (Faiss) on SGAML |
| ZooBot inference | `zoobot[pytorch]` on SGAML |
| Ellipse photometry | `py/SGA/ellipse.py` (existing SGA-2025 pipeline) |
| QA / HTML | `py/SGA/qa.py`, `py/SGA/html.py` (existing) |
| MPI execution | `bin/SGA2026-mpi` (to be written) |

SGAML environment prefix: `/global/common/software/desi/users/ioannis/SGAML`

---

## 12. Collaborators and References

- **Viraj Manwadkar** (Stanford/SLAC): shredding pipeline author, DESI DR9 dwarf catalog, pre-computed DR9 SSL embeddings. Code: https://github.com/virajvman/desi_dwarfs
- **Risa Wechsler** (Stanford/SLAC): co-PI on Manwadkar et al.
- **George Stein** (ssl-legacysurvey): MoCo v2 SSL training on 76M Legacy Survey DR9 galaxies. Code: https://github.com/georgestein/ssl-legacysurvey
- **Zoobot**: https://github.com/mwalmsley/zoobot

**Key papers:**
- Manwadkar et al. (in prep.) — "When Galaxies Fall Apart: Addressing Photometric Shredding with Color-Based Aperture Photometry in DESI DR1"
- Manwadkar et al. (companion, in prep.) — DESI DR1 dwarf galaxy catalog
- Stein et al. (2022) — ssl-legacysurvey
- Walmsley et al. (2022) — Zoobot / Galaxy Zoo DECaLS

---

## 13. Open Questions

1. **Photo-z availability for DR11**: how complete and how accurate are photo-z's over the full DR11 footprint? This determines how aggressively we can apply color-based association without DESI redshifts.
2. **Combining embedding spaces**: Manwadkar's DR9 embeddings and the SGA-2025 embeddings may use different normalizations or training runs. Verify compatibility (same backbone weights?) before merging into a single Faiss index.
3. **Cirrus at the size scale of interest**: at 15–45 arcsec, cirrus filaments can produce Tractor sources that pass all photometric cuts. Quantify the cirrus false positive rate in a test region before committing to the full pipeline.
4. **Cluster environment special-casing**: the Stage 1B (compact spheroidal) population is most important in cluster environments, where the background contamination problem is also worst (many background galaxies at the same apparent size). May need a cluster-specific selection mode.
5. **ZooBot validation**: no testing has been done yet. Before using ZooBot as a gate, validate on a sample of ~1000 known SGA and non-SGA objects spanning the 15–45 arcsec range.
6. **Scarlet at scale**: Scarlet is computationally expensive. Best used as a validation/spot-check tool for the most complex reconstructions, not as a default step.
7. **Low-surface-brightness tail**: galaxies with disconnected segmentation blobs (Manwadkar Section 5.1) are exactly the systems that are most scientifically interesting and most likely to be missing from SGA-2025. A dedicated strategy for this population (e.g., source stacking, larger segmentation apertures, or manual identification from known LSB surveys) is deferred to a future iteration.
