# UniMemo: Cross-Platform AI Context Extension

UniMemo is a browser extension and infrastructure project designed to capture, manage, and seamlessly sync conversation histories across different LLM platforms (e.g., ChatGPT, Claude, Gemini). By tracking entities and context across your AI interactions, UniMemo ensures you never lose important facts or mid-conversation corrections when switching contexts.

## Core Features

- **Cross-Platform Syncing**: Unifies your fragmented AI conversations into a single, cohesive memory stream.
- **Entity Tracking**: Automatically extracts and tracks key entities (dates, technologies, names) to ensure factual consistency.
- **Smart Compression**: Utilizes our novel CSGC algorithm to compress long conversation histories before injecting them into new context windows, saving API costs and reducing TTFT (Time-To-First-Token).

---

## 🍒 Research Highlight: CSGC & Pivot Recall

As a core part of the UniMemo infrastructure, we developed **CSGC (Context-Salience Greedy Compression)** to solve a critical issue in modern AI usage: *supersession*. When users correct themselves mid-conversation, standard compressors often discard the correction while keeping the obsolete fact.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21227878.svg)](https://doi.org/10.5281/zenodo.21227878)

**Read the full paper on Zenodo:** [CSGC: Compressing Conversation History Without Losing Corrections](https://doi.org/10.5281/zenodo.21227878)

### CSGC Pipeline
CSGC is a deterministic, three-stage pipeline:
1. **Redundancy Collapse**: Merges identical messages and detects factual corrections using `e5-small-v2` embeddings and entity tracking.
2. **Salience Scoring**: Scores messages using a weighted combination of TF-IDF, entity density, and recency, applying a 1.3× boost to corrections and a 0.5× penalty to obsolete facts.
3. **Greedy Budget Selection**: Packs the highest-scoring messages into a strict byte budget.

### Pivot Recall Metric
We introduced **Pivot Recall**, a metric that explicitly tests whether a compressor successfully retains a *corrected* fact over the obsolete fact it replaced. In our WildChat-1M benchmarks, CSGC achieved a **58.3% Pivot Recall**, compared to a catastrophic 5.5% for LLMLingua-2, all while matching its 89% semantic retention and running **4.2× faster**.

---

## Repository Structure

- `background/`, `content/`, `popup/`, `icons/`, `utils/`: The core source code for the UniMemo browser extension.
- `research/`: The Python evaluation pipeline (`CSGC_Rebuild.ipynb`) used to benchmark the CSGC algorithm against LLMLingua-2, TextRank, TF-IDF, and Recency.
- `paper/`: The complete LaTeX source code (`main.tex`) and figures for the CSGC research paper.

## Citation

If you use the UniMemo architecture, CSGC, or the Pivot Recall metric in your work, please cite our Zenodo deposit:

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
