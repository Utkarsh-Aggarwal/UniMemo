# CSGC: Context-Salience Greedy Compression

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21227878.svg)](https://doi.org/10.5281/zenodo.21227878)

This repository contains the evaluation pipeline and paper manuscript for **"CSGC: Compressing Conversation History Without Losing Corrections—Entity-Aware Supersession Detection and the Pivot Recall Metric."**

Read the full paper on Zenodo: [10.5281/zenodo.21227878](https://doi.org/10.5281/zenodo.21227878)

## Overview

Modern LLM-powered applications accumulate extensive conversation histories that must be compressed before being re-injected into limited context windows. While current approaches optimise for average semantic overlap with the original text, they ignore a critical temporal phenomenon: users and assistants routinely *correct* earlier facts during a conversation. A compressor that retains obsolete versions can degrade response accuracy.

We introduce **Context-Salience Greedy Compression (CSGC)**, a deterministic, three-stage pipeline (Redundancy Collapse, Salience Scoring, and Greedy Budget Selection) that explicitly detects and deprioritises superseded messages while promoting their corrections.

We also introduce **Pivot Recall**, a novel evaluation metric that tests whether a compressed output contains the *corrected* version of a fact rather than the one it replaced.

## Repository Structure

- `paper/`: Contains the complete LaTeX source code (`main.tex`) and all figures used in the manuscript. The references are provided inline so the file can be compiled standalone.
- `research/`: Contains the Python evaluation pipeline, primarily `CSGC_Rebuild.ipynb`, which implements the full CSGC algorithm and benchmarks it against methods like LLMLingua-2, TextRank, TF-IDF, and Recency.

## Evaluation & Results

Evaluated on 99 long-context English conversations from WildChat-1M at a 30% byte budget, our Lean variant (`CSGC_Lean`) achieves **91.2% semantic retention** (matching LLMLingua-2's 89.1% while running **4.2× faster**) and achieves a Pivot Recall of **58.3%**—compared to 50.4% for Recency-based truncation and 5.5% for LLMLingua-2.

These results suggest that perplexity-guided token distillation does not inherently preserve factual corrections, whereas lightweight entity-tracking recovers the majority of this structural signal without requiring LLM calls.

## Citation

If you use CSGC or the Pivot Recall metric in your work, please cite the Zenodo deposit:

```bibtex
@misc{aggarwal2026csgc,
  author = {Aggarwal, Utkarsh},
  title = {CSGC: Compressing Conversation History Without Losing Corrections---Entity-Aware Supersession Detection and the Pivot Recall Metric},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.21227878},
  url = {https://doi.org/10.5281/zenodo.21227878}
}
```
