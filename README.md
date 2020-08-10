# StackingSingleCellClassify

Comprehensive single-cell clustering implementations by python

## Files and functions

### `ReadData.py`

Read data from `.mat`, `.csv` and `.txt` files.

### `Utils.py`

Some tools including:

* Get colors for painting base on labels.
* Scatter with displaying labels.

### `DimensionReduction.py`

Methods of dimension reduction including:

* t-SNE
* PCA

### `Clustering.py`

Clustering methods including:

* k-means
* k-NN

### `Examples_***.py`

Examples for processing.

## Xin dataset (human islet) 1600 samples using t-SNE

perplexity = 50

![图片](pics/xin_human_islet_perp50.png)

perplexity = 5

![图片](pics/xin_human_islet_perp5.png)

This dataset file is too large to upload, please download it from:

[data-download](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE81608&format=file&file=GSE81608%5Fhuman%5Fislets%5Frpkm%2Etxt%2Egz)

[label-download](https://s3.amazonaws.com/scrnaseq-public-datasets/manual-data/xin/human_islet_cell_identity.txt)

Xin, Y. et al. RNA Sequencing of Single Human Islet Cells Reveals Type 2 Diabetes Genes. Cell Metab. 24, 608–615 (2016)

## Yang dataset (human embryo devel) 90 samples using t-SNE

perplexity = 40

![图片](pics/yang_human_embryo_devel_perp40.png)

perplexity = 5

![图片](pics/yang_human_embryo_devel_perp5.png)

Yan, L. et al. Single-cell RNA-Seq profiling of human preimplantation embryos and embryonic stem cells. Nat. Struct. Mol. Biol. 20, 1131–1139 (2013)

## Corr datasets

Human islet dataset / 60 samples / perplexity = 5

![图片](pics/corr_islet_perp5.png)

Human islet dataset / perplexity = 5 / k-NN

![图片](pics/corr_islet_perp5_knn.png)

Accuracy = 0.9667 (58/60) The class 'delta' (2 samples) are totally missed.

Human cancer dataset / 33 samples / perplexity = 5

![图片](pics/corr_hcancer_perp5.png)

Jiang, H., Sohn, L., Huang, H., & Chen, L. (2018). Single Cell Clustering Based on Cell-Pair Differentiability Correlation and Variance Analysis. (May).
