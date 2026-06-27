# 🌍 Wolof Corpus — LLMOps Pipeline

[![CI](https://github.com/SEYDINA04/wolof-corpus-llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/SEYDINA04/wolof-corpus-llmops/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-galsenai%2Fwolof__centalized__corpus-FFD21E)](https://huggingface.co/datasets/galsenai/wolof_centalized_corpus)
[![Examples](https://img.shields.io/badge/examples-606%2C456-success)](https://huggingface.co/datasets/galsenai/wolof_centalized_corpus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline **LLMOps** de construction, validation et publication d'un corpus
**wolof** centralisé pour l'entraînement de modèles de langue (LLM/ASR/MT).

Le projet agrège, déduplique et contrôle automatiquement de multiples sources
publiques (texte web, traductions, ASR, instructions LLM) puis publie un dataset
unique et versionné sur HuggingFace :
**[`galsenai/wolof_centalized_corpus`](https://huggingface.co/datasets/galsenai/wolof_centalized_corpus)**

---

## 📊 Le corpus en chiffres

| Indicateur | Valeur |
|---|---|
| Exemples | **606 456** |
| Tokens (whitespace) | **20.6 M** |
| Texte brut (UTF-8) | **107.7 MB** |
| Part de wolof (GlotLID) | **95.2 %** |
| Sources agrégées | **33** |
| Exemples multi-sources | 133 601 |
| Format | Parquet · `text` (string) + `sources` (list[string]) |

> Construit en fusionnant le corpus central existant (304 762 ex.) avec
> **17 nouveaux datasets HF** (424 119 ex.), déduplication exacte appliquée
> (+313 477 exemples nets, 0 donnée existante perdue).

---

## 🏗️ Architecture

```
pipeline.yaml ─────────────────────────► configuration unique (config as code)
                                              │
 [1] ingest ─► [2] centralize ─► [3] merge ─► [4] stats ─► [5] datacard ─► [6] validate ─► [7] publish
 datasets HF   jsonl→parquet     +corpus HF    GlotLID      README auto     QUALITY GATES    HF (versionné)
 (resumable)   dédup interne     dédup global  volumétrie                   (bloquant)
```

| # | Étape | Rôle |
|---|---|---|
| 1 | **ingest** | Ingestion des datasets HF → `jsonl` (filtrage langue GlotLID + longueur, reprise possible) |
| 2 | **centralize** | Fusion des `jsonl` → corpus ingéré (parquet, dédup interne) |
| 3 | **merge** | Fusion corpus HF + ingéré → corpus unifié (dédup globale, fusion des `sources`) |
| 4 | **stats** | Volumétrie + détection de langue (échantillon) |
| 5 | **datacard** | Génération automatique du README HF depuis les stats |
| 6 | **validate** | **Quality gates** bloquants (schéma, %wolof, taille, anti-perte…) |
| 7 | **publish** | Publication parquet + data card sur HuggingFace |

Orchestré par `src/pipeline/run.py`, exposé via le `Makefile`.
📖 Documentation détaillée : **[`docs/LLMOPS.md`](docs/LLMOPS.md)**

---

## 🛡️ Quality Gates (contrôles bloquants)

Aucune publication si un seul échoue (config dans `src/pipeline.yaml`) :

| Gate | Seuil par défaut |
|---|---|
| `schema` (colonnes/types) | `text:str`, `sources:list` |
| `min_examples` | ≥ 600 000 |
| `empty_texts` | ≤ 0 |
| `duplicates` | ≤ 0.0 % |
| `raw_text_size` | ≥ 100 MB |
| `wolof_pct` | ≥ 90 % |
| `no_hf_loss` (anti-régression) | 0 perte |

---

## 📚 Ressources & sources de données

**Datasets HF ingérés (17)** — déclarés dans `src/pipeline.yaml` :

| Catégorie | Sources |
|---|---|
| Texte natif / web | `soynade-research/FineWeb2-HQ-50k-Wolof`, `soynade-research/Wolof-Non-Standard-Orthography` |
| Traductions | `ZigZeug/Baatukaay-wolof-translated-dataset`, `bilalfaye/english-wolof-french-dataset`, `bilalfaye/wolof-english-french`, `Bassoumm/wolof-french-dictionary`, `skonteye/French-Wolof-Dataset-With-Sources`, `galsenai/english-wolof-smol-translation`, `MaroneAI/French-Wolof_Translation-Dataset`, `MaroneAI/Wolof-to-French_Translation-Dataset`, `mbaye930/wolof-arabic-parallel-corpus` |
| ASR / divers | `soynade-research/Wolof-Agri-Captions`, `vonewman/fleurs-wolof-dataset`, `michsethowusu/wolof-sentiments-corpus`, `mbaye930/WolofEntityLinking` |
| Instructions LLM | `m-a-d-i/wori-wolof-instructions`, `ngia/alpaca-data-in-wolof` |

**Outils & modèles :**
- [GlotLID](https://huggingface.co/cis-lmu/glotlid) (`cis-lmu/glotlid`) — détection de langue (`wol_Latn`)
- [`datasets`](https://github.com/huggingface/datasets), [`huggingface_hub`](https://github.com/huggingface/huggingface_hub), `pandas`, `pyarrow`
- [`uv`](https://github.com/astral-sh/uv) — gestion d'environnement, `ruff` (lint), `pytest` (tests)

---

## 🚀 Démarrage rapide

```bash
# 1. Secrets (une fois)
cp .env.example .env          # renseigner HF_TOKEN (write sur galsenai)

# 2. Dépendances
make setup                    # = cd src && uv sync --extra dev

# 3. Pipeline
make all                      # ingest → centralize → merge → stats → datacard → validate
make validate                 # quality gates seuls
make publish                  # publie sur HF (refusé si gates KO)

make help                     # toutes les commandes
```

---

## 🔄 CI/CD (GitHub Actions)

| Workflow | Déclencheur | Rôle |
|---|---|---|
| `ci.yml` | push / PR | `ruff` (lint) + `pytest` (tests) |
| `publish.yml` | manuel (`workflow_dispatch`) | pipeline complet + gates + publication |

**Publication via GitHub** : onglet *Actions* → *Publish corpus to HuggingFace*
→ *Run workflow* → taper `PUBLISH`. Pré-requis : secret `HF_TOKEN` du dépôt.

---

## 📁 Structure du projet

```
wolof_scraper/
├── Makefile                     # interface du pipeline
├── pyproject.toml · uv.lock     # projet racine
├── .env.example                 # template secrets (.env est gitignored)
├── docs/
│   └── LLMOPS.md                # documentation complète du process
├── .github/workflows/
│   ├── ci.yml                   # tests automatiques
│   └── publish.yml              # publication manuelle + gates
└── src/
    ├── pipeline.yaml            # ★ configuration unique
    ├── pipeline/                # orchestration LLMOps
    │   ├── config.py            #   chargement config + secrets
    │   ├── quality_gates.py     #   contrôles bloquants
    │   ├── datacard.py          #   génération README HF
    │   └── run.py               #   orchestrateur CLI (7 stages)
    ├── ingest_hf_datasets.py    # [1] ingestion
    ├── centralize_ingested.py   # [2] centralisation
    ├── merge_corpora.py         # [3] fusion dédupliquée
    ├── corpus_stats.py          # [4] statistiques
    └── tests/                   # tests pytest
```

---

## 🧰 Stack technique

`Python 3.12` · `uv` · `pandas` / `pyarrow` · `datasets` / `huggingface_hub`
· `fasttext` + `GlotLID` · `ruff` · `pytest` · `GitHub Actions`

---

## 👤 Auteur & affiliations

**Babacar Ndao** — *Data & AI System Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-SEYDINA04-181717?logo=github&logoColor=white)](https://github.com/SEYDINA04)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-SEYDINA04-FFD21E)](https://huggingface.co/SEYDINA04)

Réalisé dans le cadre de **[GalsenAI](https://huggingface.co/galsenai)** —
communauté sénégalaise dédiée à l'intelligence artificielle et aux langues
locales (wolof).

---

## 📄 Licence

Code sous licence [MIT](LICENSE). Le dataset publié est sous `cc-by-4.0` (voir la data card HF).
Chaque source conserve sa licence d'origine — se référer aux datasets respectifs.
