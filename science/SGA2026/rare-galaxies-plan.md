# Research Plan: Finding Rare Galaxies in SGA-2025 with SSL Embeddings

**Status:** Working draft (2026-10-02)  
**Student:** Muhammad Fatin Rajput  
**Advisor:** John Moustakas  
**Term:** Fall 2026 independent study (ends 2026-12-15)

---

## 1. Science Question

The SGA-2025 contains several hundred thousand large, well-resolved galaxies, far too many to inspect by eye. We have a 2048-dimensional self-supervised (MoCo v2) embedding of the *grz* image of each one. We want to know whether those embeddings are good enough to find rare morphologies from a handful of examples.

**Primary question.** Given a small number of known ring galaxies as queries, how efficiently does an embedding similarity search recover other ring galaxies in the SGA-2025? We measure efficiency as the fraction of known rings recovered versus the number of candidates a person has to inspect, and we compare against simple baselines.

**Secondary product.** A ranked list of the most unusual objects in embedding space, visually classified into real astrophysical oddities and data problems.

We focus on rings rather than mergers for a practical reason. Each SSL cutout is the full *group* mosaic rescaled to 152×152 pixels, so any multi-galaxy group looks like an interacting system regardless of whether it is one. Rings are a single-galaxy feature, they are easy to recognize by eye, and reference catalogs exist. Restricting the primary analysis to isolated galaxies (`GROUP_MULT == 1`) removes the group confusion entirely.

If the ring search works well and time allows, mergers and disturbed systems are a natural extension, but they are not part of the fall plan.

---

## 2. Data

All data live at NERSC under
`$CFS/desicollab/users/ioannis/SGA/2025/ssl` (see `README-ssl.txt` in
that directory) or
https://data.desi.lbl.gov/desi/users/ioannis/SGA/2025/ssl/README-ssl.txt.

| Product | File | Notes |
|---|---|---|
| Embeddings | `ssl-embeddings-dr11-south.hdf5` | `embeddings` (N, 2048), `projections` (N, 128), `sgaid`, `ra`, `dec` |
| Cutouts | `ssl-cutouts-dr11-south-chunk*.hdf5` | `images` (N, 3, 152, 152), *grz*, nanomaggies |
| Catalog | `SGA.SGA.read_sga_sample(region='dr11-south')` | sizes, shapes, magnitudes, redshifts, group information |

We start with the `dr11-south` (DECam) files only. Join the embeddings to the catalog with `SGA.ssl.load_ssl_embeddings`, which matches on `SGAID`.

Three properties of the cutouts matter for interpreting results:

1. Every mosaic is resampled to 152 pixels, so the pixel scale varies from galaxy to galaxy.
2. Missing bands are zero-filled, and *i* stands in for *r* when *r* is missing. These objects will look "unusual" for uninteresting reasons, so flag them using the catalog `BANDS` column and exclude them from the main analysis.
3. The images are in raw flux units, so apparent brightness is encoded in the embedding.

**External labels** (cross-match by position; exact catalogs to be confirmed during the literature review):

- Galaxy Zoo DESI (Walmsley et al. 2023) vote fractions for broad morphology: smooth versus featured, edge-on, bar, spiral arms, merging.
- Known ring galaxies: candidates include the Buta ring catalogs, the Galaxy Zoo 2 "ring" votes, and the ring sample from Walmsley et al. (2022).

Using the published Galaxy Zoo DESI catalog means the project does not depend on running Zoobot ourselves. Running Zoobot is a stretch goal (Section 6).

---

## 3. Working Environment

Most of the work happens in Jupyter notebooks on NERSC JupyterHub, using the SGAML kernel (`$SGAML_PREFIX/etc/activate.sh`). Notebooks are a good fit for this project because nearly every step involves looking at images.

A few habits keep notebook-based work reproducible:

- **One git repository**, with numbered notebooks that each do one job (`01-build-table.ipynb`, `02-nuisance-probes.ipynb`, ...).
- **One small Python module** (`rare.py` or similar) for anything used in more than one notebook: loading data, making an RGB image, plotting a grid of neighbors, computing recall curves. Notebooks import from it instead of copying cells.
- **Cache intermediate results to disk** (the master table, the neighbor index, label matches) so that no notebook depends on another notebook's memory.
- **Restart and run all** before calling any notebook finished, and before each weekly meeting.
- **Fix random seeds** for UMAP, train/test splits, and query selection.
- **Record visual inspection in files.** Every by-eye classification goes into a CSV with the `SGAID`, the class, and the date, never just into a notebook cell.

Nearest-neighbor searches and linear classifiers on these embeddings run on a CPU in seconds to minutes, so no batch jobs are needed for the core plan. Only the stretch goals (Zoobot inference, regenerating cutouts) need a GPU or a batch script.

---

## 4. Timeline

| Dates | Phase | Deliverable |
|---|---|---|
| Oct 5–16 | Setup, labels, reading | Master table; one-page question statement |
| Oct 19–Nov 13 | Validation experiments | Reproducible workflow (**due Nov 14**) |
| Nov 16–Dec 4 | Ring search and outlier list | Candidate catalog (**due Dec 5**) |
| Dec 7–15 | Writing | Final report and manuscript outline (**due Dec 15**) |

### Phase 1: Setup (Oct 5–16)

1. Create the repository and confirm the SGAML kernel loads the embeddings, cutouts, and catalog.
2. Build the **master table**: one row per embedded galaxy with the catalog columns we need (`SGAID`, `RA`, `DEC`, `D26`, axis ratio, magnitudes, redshift, `GROUP_MULT`, `BANDS`), a band-coverage flag, and both embedding arrays.
3. Write the image-grid function: given a list of `SGAID`s, show their RGB cutouts. Everything later depends on it.
4. Cross-match the master table to Galaxy Zoo DESI and to one or more ring catalogs. Report how many SGA galaxies have labels and how many known rings we have.
5. Read Hayat et al. (2021), Stein et al. (2022), and Walmsley et al. (2022). Write one page: what each paper did, and which of their tests we will repeat on the SGA.
6. Write a one-page statement of the question, the sample cuts, and the definition of success.

### Phase 2: Validation (Oct 19–Nov 13)

Each experiment is one notebook and produces one or two figures.

1. **A first look.** Make a UMAP of the embeddings, colored in turn by size, magnitude, color, axis ratio, and Galaxy Zoo labels. Pick ten galaxies and show their 20 nearest neighbors. This is qualitative, but it builds intuition for everything else.
2. **What do the embeddings encode?** Train a simple linear model to predict `D26`, magnitude, color, axis ratio, redshift, and band coverage from the embedding. Properties that are easy to predict are the ones that drive "similarity." If apparent size or brightness dominates, we compare neighbors only within matched bins of those properties.
3. **Do neighbors share morphology?** For each labeled galaxy, measure the fraction of its *k* nearest neighbors with the same Galaxy Zoo label (for example, barred or edge-on). Compare with a random sample matched in size and magnitude. Repeat for the 2048-d embeddings and the 128-d projections, and adopt whichever performs better.
4. **Ring recovery.** Split the known rings into query and held-out sets. Rank all galaxies by similarity to the queries and plot the fraction of held-out rings recovered versus the number of candidates inspected. Vary the number of queries (1, 10, 100). Compare three methods: nearest neighbors, a linear classifier trained on the embeddings, and a random-ranking baseline.

By Nov 14 a new user should be able to clone the repository, run the notebooks in order, and reproduce every figure.

### Phase 3: Search (Nov 16–Dec 4)

1. Before inspecting anything, agree on a written definition of "ring" (inner, outer, collisional, polar; what counts as a pseudo-ring) and a short list of classes for the inspection CSV.
2. Run the best method from Phase 2 over the full sample. Visually inspect the top-ranked candidates that are *not* in any reference catalog, in ranked order, until the yield drops.
3. Characterize the result: purity versus rank, typical false positives, and how the recovered rings are distributed in size, magnitude, and redshift compared with the known rings.
4. Outlier list: rank galaxies by distance to their *k*-th nearest neighbor, inspect the top several hundred, and sort them into classes (real and unusual, artifact, bad mosaic, bright star). The artifact classes are useful feedback for the SGA itself.

### Phase 4: Writing (Dec 7–15)

1. Final report: data, methods, results, limitations, with the figures from Phases 2 and 3.
2. A manuscript outline with a figure list, and an honest assessment of what is still needed for a paper.

---

## 5. Risks

| Risk | What we do |
|---|---|
| Embeddings mostly track apparent size, brightness, or pixel scale | Match in those properties, or regress them out. If that fails, regenerate cutouts at a fixed angular scale with `SGA2025-ssl-cutouts` (decide by the end of October) |
| Too few known rings overlap the SGA | Label a starter set by eye from nearest-neighbor grids, and use those as queries |
| Similarity search performs no better than the baselines | Report it. A careful negative result with a clear diagnosis is still a complete project |
| Visual inspection takes longer than planned | Cap the inspection at a fixed number of candidates and report purity for that set |

---

## 6. Stretch Goals

- Run Zoobot (pre-trained weights, in SGAML) on the SGA cutouts and use its predictions as a supervised baseline for the ring search. A documented, working recipe would also serve the SGA-2026 completeness plan (`completeness-plan.md`, Section 7).
- Add the `dr11-north` embeddings.
- Extend the search to mergers and tidally disturbed systems, using per-galaxy rather than per-group cutouts.

---

## 7. First Two Weeks: Checklist

- [ ] Repository created and shared
- [ ] SGAML Jupyter kernel working; embeddings and catalog load
- [ ] Master table written to disk
- [ ] Image-grid function working
- [ ] Galaxy Zoo DESI and ring-catalog cross-matches done, with counts
- [ ] UMAP and nearest-neighbor grids for ten example galaxies
- [ ] One-page reading summary and one-page question statement

---

## 8. References

- Hayat et al. 2021, ApJL, 911, L33 (self-supervised representations for astronomical images)
- Stein et al. 2022 (ssl-legacysurvey; https://github.com/georgestein/ssl-legacysurvey)
- Walmsley et al. 2022, MNRAS, 509, 3966 (Galaxy Zoo DECaLS)
- Walmsley et al. 2022 (practical morphology tools from deep representations; ring search)
- Walmsley et al. 2023 (Galaxy Zoo DESI; Zoobot)
- Moustakas et al. 2023, ApJS, 269, 3 (SGA-2020)
