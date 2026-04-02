# Concept Pruning — FP2 / Independent Coursework 2 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xai-concept-pruning-mitoses-ug7bxicybwkp7mu5yj8pwa.streamlit.app/)

**Classification based on neurally learned concepts**
HTW Berlin · Master Applied Informatics · Independent Coursework 2
Ilona Eisenbraun · October 2022 · Supervisor: Prof. Dr. rer. nat. Peter Hufnagl

> Full report: [`docs/ICW_report.pdf`](docs/ICW_report.pdf)

---

## Overview

This project investigates which **concepts** a VGG-based mitosis classifier
(developed at CBMI, HTW Berlin) learns and relies upon for its decisions.
It is the direct continuation of two earlier courseworks:

| Coursework                  | Topic                                                           |
| --------------------------- | --------------------------------------------------------------- |
| ICW 1 — LRP Pruning        | Network pruning guided by Layerwise Relevance Propagation (LRP) |
| ICW 1 — ProtoPNet          | Interpretable classification via learned prototypes             |
| **ICW 2 (this repo)** | **Concept attribution and concept-level pruning via CRP** |

The central tool is **Concept Relevance Propagation (CRP)** [Achtibat et al., 2022],
implemented through the `zennit-crp` Python package. CRP extends LRP to attribute
network decisions not just to input pixels but to individual convolutional filters
(concepts), enabling a layer-by-layer view of what the network has learned.

---

## Research Context

### Problem

Neural networks for medical image classification — here: detecting mitotic figures
in histopathology slides — are black boxes. Understanding *which internal concepts*
drive a positive prediction ("mitosis") supports both model validation and pruning.

### Approach

Three complementary attribution strategies are implemented:

1. **Latent (layer-crossing) attribution** — For each layer, the top-5 concept
   encoders across all training samples are identified via a voting mechanism.
   The result is a `concept_atlas` dictionary mapping each layer to its most
   active concept IDs and their vote counts.
2. **Image-region-specific attribution** — A 100×100 pixel centre mask is applied
   to the input heatmap (motivated by mitoses appearing predominantly in the image
   centre). This provides a spatially constrained view of which concepts respond to
   the actual mitotic region.
3. **Attribution fragmentation (sub-concept decomposition)** — For each layer's
   dominant concept, back-propagation is restarted from that concept to the
   preceding layer (`record_layer=[prev_layer]`), revealing the sub-concepts that
   compose each higher-level concept.

### Key Findings

- **Middle and higher layers are decisive.** `features.25`, `classifier.0`, and
  `classifier.3` contain concepts with the highest percentage relevance (up to
  ~1.06 for concept 370 in `features.25`). Lower layers show near-zero per-concept
  relevance, yet encode consistent patterns across samples.
- **Lower layers are not irrelevant — they are uniform.** Across all three
  attribution methods, the same concept IDs recur with maximum vote counts (40/40
  samples) in lower layers (`features.0`, `features.4`). This indicates stable,
  low-level texture encodings rather than meaningful discriminative concepts.
- **Concepts are hierarchically decomposable.** The fragmentation analysis shows
  that each layer's dominant concepts trace back to a coherent set of sub-concepts
  in the preceding layer. For example, the top concepts in `features.22` decompose
  predictably into a cluster of adjacent filter indices in `features.18`.
- **Spatial consistency.** The centre-masked local analysis recovers a concept
  atlas closely aligned with the global latent analysis, confirming that the
  network's mitosis-specific representations are centred on the actual mitotic
  structures in the input image.

### Positioning in the XAI Landscape

CRP sits at the intersection of **attribution methods** (explaining a specific
prediction) and **concept-based explanations** (revealing what the network has
learned globally). Compared to TCAV or Net Dissection, CRP is gradient-based and
therefore directly compatible with LRP composites, making it straightforward to
apply to the same VGG architecture used in the LRP pruning coursework (ICW 1).

The observation that high-level classifier concepts carry disproportionate relevance
aligns with findings in the broader XAI literature: later layers encode task-specific
abstractions while earlier layers encode low-level features that are shared across
classes. This has direct implications for pruning strategies — concept-level pruning
in lower layers risks removing features relied upon by multiple classes, whereas
pruning in `features.22` and above targets genuinely class-specific encodings.

---

## Project Structure

```
.
├── main.py                      Entry point: data loading, fine-tuning, evaluation loop
├── requirements.txt
├── modules/
│   ├── concept_attribution.py   Latent, local and decomposition attribution methods
│   ├── concept_pruning.py       Mask-based pruning of selected concept filters
│   ├── concept_visualization.py Heatmap generation and composite image rendering
│   ├── lamb.py                  LAMB optimiser (self-contained, no external path required)
│   └── utils.py                 Sparsity measurement, attribution variable initialisation
├── analysis/
│   ├── CocoX.ipynb
│   ├── model_crp.ipynb
│   └── model_crp_2.ipynb
├── outputs/
│   ├── true_attributions/       Heatmap outputs for the mitosis (True) class
│   └── false_attributions/      Heatmap outputs for the non-mitosis (False) class
├── docs/
│   ├── ICW_report.pdf           Full coursework report (German)
│   ├── seminar.tex              LaTeX source (compiles with pdfLaTeX + BibTeX)
│   └── images/                  Figures used in the report
└── rag/                         XAI Paper Assistant (see below)
    ├── app.py
    ├── pipeline.py
    ├── explain.py
    ├── xai_paper_assistant.ipynb
    ├── requirements_rag.txt
    └── papers/
```

---

## XAI Paper Assistant (RAG)

An explainable RAG pipeline built on top of this research — query the CRP paper,
the coursework report, and related XAI literature and get grounded answers with
source attribution.

Most RAG demos show *which* chunks were retrieved. This one shows *why*:

| Feature                          | What it answers                                            |
| -------------------------------- | ---------------------------------------------------------- |
| **Embedding space (UMAP)** | Where does the query land relative to all document chunks? |
| **Sentence heatmap**       | Which part of each retrieved chunk is actually relevant?   |
| **Keyword overlap**        | What specific concepts connect the query to this chunk?    |

The embedding model is multilingual (`paraphrase-multilingual-MiniLM-L12-v2`),
so English queries correctly retrieve German-language chunks from the report.

**Live app:** [xai-concept-pruning-mitoses.streamlit.app](https://xai-concept-pruning-mitoses-ug7bxicybwkp7mu5yj8pwa.streamlit.app/)

**Run locally:**

```bash
pip install -r rag/requirements_rag.txt
export GROQ_API_KEY=your_key   # free at console.groq.com
streamlit run rag/app.py
```

Papers are downloaded automatically at startup. Delete `rag/chroma_db/` to trigger a re-index after adding new PDFs.

**Stack:** LlamaIndex · ChromaDB · PyMuPDF · UMAP · Groq (llama-3.1-8b-instant) · Streamlit

---

## Installation

```bash
pip install -r requirements.txt
```

The model weights (`state_dict.pth`) and dataset
(`examples_ds_from_MP.train.HTW.train/`) are not included in this repository.
Place both in the project root before running.

```bash
python main.py
```

---

## Technical Upgrades (modernize branch)

The original codebase targeted Python 3.8 / PyTorch 1.12 / Pillow 9.2.
The `modernize` branch brings it up to the current stack:

| File                                 | Change                                                                                 | Reason                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `__main__.py` → `main.py`       | Rename; replace `import imp` + hardcoded path with `from modules.lamb import Lamb` | `imp` removed in Python 3.12                            |
| `modules/lamb.py`                  | New file — self-contained LAMB optimiser                                              | Eliminates dependency on absolute path to another machine |
| `main.py`                          | `softmax(..., dim=1)`                                                                | Required since PyTorch 1.5; previously raised warning     |
| `main.py`                          | `torch.load(..., weights_only=False)`                                                | Explicit flag required in PyTorch 2.6+                    |
| `modules/utils.py`                 | `collections.abc.Mapping`                                                            | `collections.Mapping` removed in Python 3.10            |
| `modules/concept_visualization.py` | `Image.LANCZOS`                                                                      | `Image.ANTIALIAS` removed in Pillow 10.0                |
| `modules/concept_visualization.py` | `font.getbbox()`                                                                     | `font.getsize()` removed in Pillow 10.0                 |
| `modules/concept_visualization.py` | Remove `from pylab import savefig`                                                   | Deprecated matplotlib interface; unused                   |
| `modules/concept_visualization.py` | `next(iter(dict.keys()))`                                                            | `dict.keys()[0]` not valid in Python 3                  |
| `modules/concept_pruning.py`       | `device=module.weight.device`                                                        | Replaces hardcoded `device='cuda:0'`                    |
| All modules                          | Updated to `modules.*` imports                                                       | Reflects new directory structure                          |

### Directory restructure

```
Before                        After
──────────────────────────    ──────────────────────────
concept_attribution.py   →    modules/concept_attribution.py
concept_pruning.py       →    modules/concept_pruning.py
concept_visualization.py →    modules/concept_visualization.py
utils.py                 →    modules/utils.py
__main__.py              →    main.py
CocoX.ipynb              →    analysis/CocoX.ipynb
model_crp*.ipynb         →    analysis/
false_attributions/      →    outputs/false_attributions/
true_attributions/       →    outputs/true_attributions/
```

---

## Dependencies

| Package       | Original version | Minimum (modernized) |
| ------------- | ---------------- | -------------------- |
| torch         | 1.12.0           | ≥ 2.0.0             |
| torchvision   | —               | ≥ 0.15.0            |
| zennit        | 0.4.7            | ≥ 0.4.6             |
| zennit-crp    | 0.4.2            | ≥ 0.6.0             |
| Pillow        | 9.2.0            | ≥ 10.0.0            |
| numpy         | 1.22.4           | ≥ 1.24.0            |
| opencv-python | 4.6.0            | ≥ 4.7.0             |
| scikit-learn  | —               | ≥ 1.2.0             |
| pandas        | —               | ≥ 2.0.0             |
| matplotlib    | —               | ≥ 3.7.0             |
| seaborn       | —               | ≥ 0.12.0            |

---

## References

- Achtibat et al., *"From 'Where' to 'What': Towards Human-Understandable Explanations through Concept Relevance Propagation"*, arXiv:2206.03208, 2022.
- Achtibat, Dreyer & Lapuschkin, *Zennit-CRP: An extension to Zennit using CRP*, 2022.
- Anders et al., *"Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit"*, CoRR abs/2106.13200, 2021.
- Yeom et al., *"Pruning by Explaining: A Novel Criterion for Deep Neural Network Pruning"*, CoRR, 2019.
