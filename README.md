<div align="center">

# 🧠 UniMemo

### *One memory, across every AI.*

**UniMemo** is a browser extension that gives your AI assistants a unified, persistent memory — so your conversations with ChatGPT, Claude, and Gemini finally talk to each other.

<br/>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21227878.svg)](https://doi.org/10.5281/zenodo.21227878)
![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Chrome%20Extension-yellow)
![Status](https://img.shields.io/badge/status-active%20research-brightgreen)

</div>

---

## The Problem

You spend an hour with ChatGPT debugging a system architecture. Then you switch to Claude for a second opinion — and have to start from scratch. You've already corrected yourself twice in that session ("actually, use MySQL not Postgres") but a new chat window means all of that context is lost. Again.

Every LLM platform has its own isolated memory. Factual corrections get dropped. Context evaporates. You repeat yourself constantly.

**UniMemo fixes this.**

---

## What UniMemo Does

UniMemo runs quietly in your browser, capturing your conversations across platforms and building a structured context layer on top of them.

| Capability | Description |
|---|---|
| 🔗 **Cross-Platform Context** | Scrapes and unifies conversations from ChatGPT, Claude, and Gemini |
| 🧩 **Entity Tracking** | Extracts and tracks key entities — names, dates, decisions, code choices |
| ⚡ **Smart Compression** | Injects only what matters into your next session, within token budget |
| 🔄 **Supersession Awareness** | Knows when you've corrected yourself and preserves the *updated* fact |

---

## How It Works

UniMemo is built around three core components:

**1. Platform Scrapers** (`content/`)
Lightweight content scripts for ChatGPT, Claude, and Gemini that extract conversation turns in real time without storing any data externally.

**2. Service Worker** (`background/service-worker.js`)
A persistent background process that orchestrates context collection, entity resolution, and compression across tabs.

**3. CSGC Compression Engine** (`utils/compression.js`)
Our custom compression algorithm (described below) that intelligently selects which parts of your history to carry forward — prioritising corrections over obsolete facts, entities over filler, and recency where it matters.

---

## 📄 Research: CSGC & The Pivot Recall Metric

As part of building UniMemo's compression layer, we identified a flaw in *all* existing prompt compression methods: **none of them know when a fact has been corrected**.

We built **CSGC (Context-Salience Greedy Compression)** to solve this, and published the findings on Zenodo.

> 📖 **[CSGC: Compressing Conversation History Without Losing Corrections](https://doi.org/10.5281/zenodo.21227878)**  
> Entity-Aware Supersession Detection and the Pivot Recall Metric  
> *Utkarsh Aggarwal — Zenodo, 2026*  
> DOI: `10.5281/zenodo.21227878`  
> *(arXiv submission pending endorsement)*

### Key Findings

Evaluated on 99 long-context conversations from WildChat-1M at a 30% byte budget:

| Method | Semantic Retention | Pivot Recall | Runtime |
|---|---|---|---|
| **CSGC Lean** | **91.2%** | **58.3%** | 795 ms |
| LLMLingua-2 | 89.1% | 5.5% | 3,371 ms |
| Recency | 85.4% | 50.4% | ~0 ms |
| TF-IDF | 93.0% | 43.0% | 10 ms |

**CSGC runs 4.2× faster than LLMLingua-2** and dramatically outperforms it on Pivot Recall — the metric that actually matters when your conversation has evolved.

### What is Pivot Recall?

Standard metrics measure how similar the compressed output is to the original. They can't tell you whether the compressor kept *"use MySQL"* or *"use Postgres"*. **Pivot Recall** does. It explicitly tests whether the compressed context preserves a factual correction over the obsolete belief it replaced. UniMemo's CSGC achieves 58.3% — the highest of any method tested.

The research pipeline lives in `research/CSGC_Rebuild.ipynb`.

---

## Repository Structure

```
UniMemo/
├── background/
│   └── service-worker.js       # Orchestrates context & compression
├── content/
│   ├── chatgpt-scraper.js      # ChatGPT conversation extractor
│   ├── claude-scraper.js       # Claude conversation extractor
│   └── gemini-scraper.js       # Gemini conversation extractor
├── popup/
│   ├── popup.html              # Extension popup UI
│   ├── popup.css               # Popup styles
│   └── popup.js                # Popup interaction logic
├── utils/
│   ├── compression.js          # CSGC compression engine
│   └── context-formatter.js    # Context formatting for injection
├── research/
│   └── CSGC_Rebuild.ipynb      # Full evaluation pipeline
└── paper/
    ├── main.tex                # LaTeX manuscript (self-contained)
    └── figures/                # All figures used in the paper
```

---

## Citation

If you build on UniMemo or use the Pivot Recall metric in your research, please cite:

```bibtex
@misc{aggarwal2026csgc,
  author    = {Aggarwal, Utkarsh},
  title     = {CSGC: Compressing Conversation History Without Losing
               Corrections---Entity-Aware Supersession Detection and
               the Pivot Recall Metric},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21227878},
  url       = {https://doi.org/10.5281/zenodo.21227878}
}
```

---

<div align="center">
Built by <a href="mailto:utkarshaggarwal06@gmail.com">Utkarsh Aggarwal</a>
</div>
