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
