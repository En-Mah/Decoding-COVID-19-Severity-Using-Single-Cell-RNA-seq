# Single-Cell RNA-seq Analysis of COVID-19 PBMCs 

This repository contains the first part of an end-to-end single-cell RNA sequencing (scRNA-seq) pipeline applied to PBMC samples from COVID-19 patients.

The dataset includes four clinical groups:

- Healthy
- Mild
- Severe
- Critical

---

# 📊 Dataset Overview

**Dataset:** GSE293707 (NCBI GEO)  
**Tissue:** PBMC (Peripheral Blood Mononuclear Cells)  
**Raw size:** 72,809 cells × 36,602 genes  
**Samples (pooled):**

- GSM8888963 → Healthy
- GSM8888964 → Mild
- GSM8888965 → Severe
- GSM8888966 → Critical

Important note:  
Each severity corresponds to only **one pooled sample**, which limits biological replication and requires careful interpretation in downstream steps.

---

# Phase 1 — Quality Control (QC) and Filtering

The first step was to assess and remove low-quality cells.

For each cell, we computed:

- `n_genes`: number of detected genes  
- `n_counts`: total UMI counts  
- `mito_percent`: percent of mitochondrial gene expression  

Mitochondrial reads are commonly used as a quality indicator because dying or stressed cells tend to show abnormally high mitochondrial expression.

---

## Figure 1 — Relationship between n_counts and n_genes (colored by mito%)

![QC Scatter](Figures/output-2.png)

### Interpretation

This scatter plot helps visualize the structure of the dataset and the QC thresholds:

- Most cells follow the expected curve: higher `n_counts` → higher `n_genes`
- A small cluster appears at low gene counts and low UMI counts
- These low-quality cells show higher mitochondrial percentage (bright colors)

This is a typical sign of stressed or dying cells, which should be removed.

---

## Filtering thresholds

Based on the QC patterns, the following filters were applied:

- 300 < `n_genes` < 4000  
- 800 < `n_counts` < 30,000  
- `mito_percent` < 15%

After filtering:

- 63,736 cells remained  
- Genes detected in fewer than 3 cells were removed  
- Final matrix: 63,736 cells × 25,057 genes  

This filtering step is important because low-quality cells can distort downstream PCA and create artificial clusters.

---

# Phase 2 — Normalization and Feature Selection

After QC filtering, normalization was applied to remove technical effects such as sequencing depth.

Steps included:

1. Total-count normalization (`target_sum = 1e4`)
2. Log transformation
3. Scaling (mean = 0, variance = 1)

---

## Highly Variable Genes (HVGs)

Highly Variable Genes (HVGs) were selected using the **Seurat v3 method**, with:

- batch key = `gsm_id`
- top HVGs = 3,000 genes

This is an important detail because it ensures the selected HVGs represent biological variation rather than sample-specific technical artifacts.

---

# Phase 3 — PCA (Dimensionality Reduction)

Principal Component Analysis (PCA) was applied using the top 3,000 HVGs.

The goal of PCA is to reduce the dataset from thousands of genes into a smaller number of components that capture most of the variation.

---

## Figure 2 — PCA Variance Ratio (log-scale)

![Variance Ratio Log](Figures/output-3.png)

### Interpretation

This plot shows:

- A sharp drop in variance explained after the first few PCs
- PC1 captures the largest amount of variation
- The curve begins flattening after ~15 PCs

This suggests most meaningful structure is captured early.

---

## Figure 3 — PCA Variance Ratio (linear-scale)

![Variance Ratio Linear](Figures/output-4.png)

### Interpretation

This plot confirms the same idea more clearly:

- The first ~10 PCs explain most of the signal
- After ~15–20 PCs, additional PCs contribute only small gains

Based on both plots, **20 PCs** were selected for downstream analysis.

---

# Phase 4 — PCA Visualization and Global Data Structure

After PCA, we visualized the data in 2D using PC1 and PC2 to understand global structure and how severity groups relate to each other.

---

## Figure 4 — PCA colored by clinical severity

![PCA Severity](Figures/output-5.png)

### Interpretation

Several important observations can be made:

- The PBMC dataset shows the typical "L-shaped" structure in PCA space
- Severity groups strongly overlap, meaning there is no trivial separation
- However, a gradual shift can be observed from Healthy to more severe groups

This suggests that severity effects exist, but are subtle and mixed with cell-type-driven variation.

---

## Figure 5 — PCA colored by GSM sample ID

![PCA GSM](Figures/output-6.png)

### Interpretation

This plot highlights a key experimental limitation:

- Each severity corresponds to exactly one pooled sample
- Some structure may reflect sample-specific or donor-specific variation

This is important because it means later machine learning models may accidentally learn sample identity rather than disease biology if validation is not designed carefully.

---

## Figure 6 — PCA colored by binary condition (Healthy vs Disease)

![PCA Binary](Figures/output-7.png)

### Interpretation

- Healthy and Disease cells overlap heavily
- This confirms that disease signal is not cleanly separable in an unsupervised embedding
- More detailed downstream analysis (clustering, cell-type annotation, supervised learning) is required to detect severity-related transcriptional patterns

---

By the end of Phase 4:

- Low-quality cells were removed using QC filtering
- Data was normalized and scaled
- HVGs were selected in a batch-aware manner
- PCA captured the main structure of PBMC variation
- Severity effects were visible only as subtle shifts, not as clean clusters

This sets the foundation for the next stage of the pipeline: clustering and cell-type identification.







---

# Phase 5 — Clustering and Cluster-Level Exploration

After PCA, the next step was to group cells into biologically meaningful clusters.

To do this, we constructed a k-nearest-neighbor graph in PCA space and applied **Leiden clustering**.

Leiden clustering is commonly used in scRNA-seq because it:

- Works well on large graphs
- Detects communities (cell populations)
- Produces stable clusters compared to older algorithms

---

## 5.1 Testing different Leiden resolutions

Leiden clustering depends strongly on the **resolution parameter**.

- Low resolution → fewer clusters (broad populations)
- High resolution → more clusters (finer subtypes)

---

## Figure 7 — Leiden clustering across multiple resolutions

![Leiden Resolutions](Figures/output-8.png)

### Interpretation

This figure shows clustering results for:

- Resolution 0.1
- Resolution 0.2
- Resolution 0.4
- Resolution 0.6
- Resolution 0.8

Key observations:

- At **0.1**, only a few large clusters exist (too coarse).
- At **0.8**, many clusters appear (likely over-splitting).
- **0.4** provides a good balance:
  - major PBMC populations are separated
  - clusters remain large enough for downstream analysis
  - biological structure stays clean and interpretable

Based on this comparison, **resolution = 0.4** was selected as the final clustering resolution.

---

## 5.2 Final clustering (resolution = 0.4)

---

## Figure 8 — Final Leiden clustering (res = 0.4)

![Final Leiden](Figures/output-10.png)

### Interpretation

This plot shows the final Leiden clustering result:

- 18 clusters were detected (0–17)
- The clusters form clear groups in embedding space
- Several clusters are large (major immune populations)
- Several clusters are small (rare populations)

This cluster structure forms the foundation for:

- cell-type annotation
- cluster composition analysis
- downstream machine learning (cell-type specific modeling)

---

## 5.3 Cluster composition by condition (Healthy vs Disease)

To understand whether disease changes immune composition, we calculated for each cluster:

- fraction of cells coming from Disease
- fraction of cells coming from Healthy

---

## Figure 9 — Cluster composition by condition

![Cluster Composition](Figures/output-9.png)

### Interpretation

This stacked bar plot shows:

- Each bar is one Leiden cluster
- Blue = Disease
- Orange = Healthy

Important patterns:

- Some clusters are almost entirely Disease (blue-dominant)
- Some clusters are enriched for Healthy cells
- Many clusters contain a mixture

This suggests that COVID-19 affects immune composition, but not uniformly across all cell populations.

It also supports the idea that severity-related transcriptional changes may be **cell-type specific**, not global.

---

# Phase 6 — Marker Genes and Cell-Type Annotation

After clustering, we performed cluster annotation using marker genes.

This is one of the most important steps in scRNA-seq analysis, because:

- clusters alone are not meaningful without biological labels
- marker genes allow us to identify known immune cell types

Marker gene analysis was performed in two complementary ways:

1. **Ranked marker genes per cluster**  
2. **Dot plot of canonical immune markers across clusters**

---

## 6.1 Ranked marker genes per cluster

For each cluster, we identified genes that are highly expressed compared to the rest of the dataset.

---

## Figure 10 — Top marker genes for each cluster (cluster vs rest)

![Marker Ranking](Figures/output-11.png)

### Interpretation

Each panel shows:

- one cluster compared against all other cells
- top genes ranked by differential expression score

This figure provides the first biological clues for annotation.

Examples of recognizable marker patterns:

- **Monocyte markers:** LST1, S100A8, S100A9, FCN1, LGALS3  
- **T cell markers:** TRAC, IL7R, LTB  
- **Cytotoxic/NK markers:** NKG7, GNLY, GZMB  
- **B cell markers:** MS4A1, CD79A  
- **Plasma markers:** MZB1, JCHAIN  
- **Megakaryocyte markers:** PPBP  

This ranking is especially useful for identifying rare clusters.

---

## 6.2 Dot plot of canonical marker genes across clusters

To confirm cluster identity more clearly, we used a dot plot.

Dot plots summarize two key properties:

- **Dot size:** fraction of cells expressing the gene  
- **Dot color:** average expression level  

---

## Figure 11 — Dot plot of marker genes across clusters

![Marker Dotplot](Figures/output-12.png)

### Interpretation

This figure makes annotation more robust, because:

- it confirms marker expression patterns across all clusters
- it allows detection of mixed clusters
- it helps validate whether clusters correspond to known PBMC populations

For example:

- clusters with high TRAC/IL7R are likely CD4+ T cells  
- clusters with high NKG7/GNLY are NK or cytotoxic T cells  
- clusters with high MS4A1/CD79A are B cells  
- clusters with high LST1/S100A8/LYZ are monocytes  

---

## 6.3 Cluster similarity using heatmap

To check whether clusters are transcriptionally similar, we generated a heatmap based on gene expression.

---

## Figure 12 — Cluster heatmap (global expression patterns)

![Cluster Heatmap](Figures/output-13.png)

### Interpretation

This heatmap shows:

- rows: genes
- columns: clusters (ordered by hierarchical clustering)
- similar clusters appear closer together

This confirms:

- clusters form groups (for example, lymphoid vs myeloid)
- small clusters often attach to larger immune families
- the clustering is biologically coherent

---

## 6.4 Cluster similarity using a second marker visualization

We also generated an additional marker-based cluster overview.

---

## Figure 13 — Marker expression overview (violin/dot-style across clusters)

![Marker Overview](Figures/output-14.png)

### Interpretation

This visualization reinforces:

- which clusters share immune marker signatures
- which clusters represent rare populations
- which clusters are likely subtypes rather than major populations

It also serves as a second confirmation step before assigning final cell-type labels.

---

By the end of Phase 6:

- Leiden clustering was applied and optimized across resolutions
- Resolution 0.4 was selected as the best trade-off
- 18 clusters were detected
- Cluster composition was analyzed (Healthy vs Disease)
- Marker genes were computed per cluster
- Canonical markers were visualized using dot plots and heatmaps
- Clusters were prepared for final cell-type labeling

These steps provide the biological foundation needed for the next phases:
- cell-type level differential expression
- machine learning modeling per cell type
- severity prediction and generalization testing

