# Decoding-COVID-19-Severity-Using-Single-Cell-Transcriptomics

### From Raw Data to Predictive Modeling

This project presents a complete analysis of single-cell RNA sequencing (scRNA-seq) data from COVID-19 patients.

The goal of this work was not only to analyze the biological differences between disease severities, but also to carefully build and evaluate machine learning models — while avoiding common mistakes like data leakage.

The project combines:

* 🔬 Single-cell preprocessing and quality control
* 🧬 Cell type annotation and biological interpretation
* 📊 Differential expression analysis
* 🤖 Supervised machine learning models
* ⚠️ Careful validation design to prevent misleading results

📄 The full detailed report is included in this repository:
See `Report.pdf` 

---

# 📌 Project Objective

The main objective of this project was:

> To determine whether gene expression patterns in immune cells (PBMCs) can predict COVID-19 severity.

We wanted to answer:

* Do different severities (Mild, Severe, Critical) show distinct transcriptomic signatures?
* Are these signatures consistent across immune cell types?
* Can machine learning models generalize to unseen severity groups?
* How much can data leakage inflate model performance?

---

# 📂 Dataset Description

**Dataset:** GSE293707
**Source:** NCBI Gene Expression Omnibus (GEO)

This dataset contains single-cell RNA-seq data from:

* Healthy controls
* Mild COVID-19
* Severe COVID-19
* Critical COVID-19

### Dataset Statistics

* 72,809 cells (before filtering)
* 36,602 genes
* 4 pooled samples (GSM IDs):

  * GSM8888963 → Healthy
  * GSM8888964 → Mild
  * GSM8888965 → Severe
  * GSM8888966 → Critical

### Important Limitation ⚠️

Each severity group contains **only one biological sample** (pooled donors).

This means:

* We do not have true biological replicates.
* Statistical testing is limited.
* There is risk of donor-specific bias.
* Careful model validation is extremely important.

---

# 🔬 Phase 1 — Single-Cell RNA-seq Analysis Pipeline

This phase focused on cleaning, organizing, and understanding the biological structure of the dataset.

---

## 1️⃣ Quality Control (QC)

For each cell, we calculated:

* Number of detected genes (`n_genes`)
* Total UMI counts (`n_counts`)
* Percentage of mitochondrial genes (`mito_percent`)

Cells were filtered using:

* 300 < n_genes < 4000
* 800 < n_counts < 30000
* mito_percent < 15%

After filtering:

* 63,736 high-quality cells remained
* 25,057 genes remained

This step removed low-quality cells and potential doublets.

---

## 2️⃣ Normalization & Feature Selection

To remove technical bias caused by sequencing depth:

* Data was normalized (target_sum = 1e4)
* Log-transformed
* Scaled to zero mean and unit variance

Highly Variable Genes (HVGs) were selected using the Seurat v3 method.

* Top 3,000 HVGs used for downstream analysis

This ensures that downstream models focus on biological variation rather than technical noise.

---

## 3️⃣ Dimensionality Reduction

Principal Component Analysis (PCA) was applied.

* First 20 principal components selected
* Captured most biological variation
* Reduced noise and dimensionality

PCA plots showed:

* Gradual shifts between severity groups
* No perfect separation (which is expected in real biological data)

---

## 4️⃣ Clustering

Leiden clustering was performed at multiple resolutions.

Resolution 0.4 was selected because:

* Clusters were biologically meaningful
* Cell populations were not over-fragmented
* Cluster sizes remained large enough for modeling

---

## 5️⃣ Cell Type Annotation

Clusters were annotated using canonical marker genes.

Identified major immune populations:

* CD14+ Monocytes
* FCGR3A+ Monocytes
* CD4+ T cells
* CD8+ T cells
* NK cells
* B cells
* Plasma cells
* Dendritic cells
* Megakaryocytes

---

## 6️⃣ Differential Expression Analysis

Differential expression was used to identify genes associated with each severity group.

However:

Because each condition has only one sample,
formal statistical inference is not reliable.

Therefore:

* DEGs were used as **feature selection for ML**
* Not as definitive biological conclusions

---

# 🤖 Phase 2 — Machine Learning Modeling

The second phase focused on predicting disease severity.

---

# 🚨 A Critical Lesson: Data Leakage

Two modeling strategies were tested.

---

## ❌ Approach 1 — Leakage-Affected

In the first attempt:

* Differentially expressed genes were selected using the full dataset.
* Both training and test cells influenced feature selection.

Result:

* All models achieved AUC > 0.95
* Nearly identical performance across models
* Unrealistically high and suspicious results

Problem:

This is classic **data leakage**.

The model indirectly “saw” the test data during feature selection.

These results are not trustworthy.

---

## ✅ Approach 2 — Corrected Strategy

To fix this:

* Strict train/test separation enforced
* Leave-One-Severity-Out (LOSO) cross-validation used
* Healthy cells split into two disjoint subsets
* All models trained on identical splits
* Neural network included internal validation

This significantly improved methodological rigor.

Performance dropped slightly — which is expected — but remained strong.

---

# 🧠 Models Evaluated

1. Linear Discriminant Analysis (LDA)
2. Logistic Regression
3. Linear SVM
4. Advanced Neural Network (ResMLP-based MLP)

---

# 📊 Results Summary

## Cell-Type Level Results

* Monocytes, NK cells, and T cells showed very strong signals (AUC ≈ 0.98–0.99)
* Cycling cells benefited more from neural networks
* Linear models performed nearly as well as deep learning

This suggests:

The disease signal is largely linearly separable.

---

## Severity-Level Generalization

Using LOSO validation:

* Models generalized well to unseen severities
* Mild, Severe, and Critical all showed stable performance
* Indicates conserved disease-associated transcriptional signatures

---

# 🏆 Final Model Ranking

1️⃣ Advanced Neural Network (highest overall AUC)
2️⃣ Logistic Regression (best balance of performance + interpretability)
3️⃣ Linear SVM
4️⃣ LDA

---

# 🎯 Practical Conclusion

Although the neural network had the highest AUC:

Logistic Regression is the most practical model because:

* Nearly identical performance
* Much easier to interpret
* Computationally efficient
* Transparent gene weight interpretation

This aligns with published benchmarking studies (PMID: 35281805).

---

# 🧬 Biological Insights

* COVID-19 induces strong immune-related transcriptomic shifts.
* Myeloid cells (Monocytes, DCs) carry strong disease signals.
* NK cells also show robust signatures.
* The disease signature is conserved across severities.

This suggests potential for generalized immune-based diagnostic models.

---

# ⚠️ Limitations

* Only one biological sample per condition
* Potential residual feature-level leakage
* Healthy sample split across train/test
* No external validation dataset

Results should be interpreted cautiously.

---

# 🚀 Future Improvements

* Add more biological replicates
* Use pseudo-bulk modeling
* External validation cohort
* ElasticNet regularization
* More robust feature selection inside cross-validation folds

---

# 🛠️ Tools & Libraries

* Python
* Scanpy
* scikit-learn
* PyTorch
* NumPy
* Matplotlib

---
