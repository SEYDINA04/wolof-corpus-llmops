# LLMOps — Pipeline du corpus wolof centralisé

Ce document explique **comment fonctionne** le pipeline de données et **comment
l'opérer**. Objectif : un processus reproductible, sécurisé et automatisé pour
construire et publier le dataset `galsenai/wolof_centalized_corpus`.

---

## 1. Qu'est-ce que le « LLMOps » ici ?

Le LLMOps applique les principes DevOps/MLOps à la chaîne de données qui nourrit
un LLM. Concrètement, on garantit 5 propriétés :

| Principe | Comment c'est appliqué dans ce projet |
|---|---|
| **Config as code** | Tout est déclaré dans `src/pipeline.yaml` (datasets, seuils, cibles). Aucune valeur en dur. |
| **Quality gates** | `pipeline/quality_gates.py` bloque toute publication non conforme (schéma, % wolof, taille, anti-perte). |
| **Reproductibilité** | Étapes idempotentes/resumables, `uv.lock`, modèle GlotLID figé. |
| **Sécurité** | Secrets hors du repo : `.env` en local, *secret GitHub* en CI. |
| **CI/CD** | GitHub Actions : tests à chaque push, publication manuelle contrôlée. |

---

## 2. Architecture du pipeline

```
pipeline.yaml  ─────────────► config unique (datasets, seuils, repo HF)
                                   │
 [1] ingest ─► [2] centralize ─► [3] merge ─► [4] stats ─► [5] datacard ─► [6] validate ─► [7] publish
 datasets HF   jsonl→parquet     +corpus HF    GlotLID      README auto     QUALITY GATES    HF (versionné)
 (resumable)   dédup interne     dédup global  volumétrie                   (bloquant)
```

| # | Stage | Script / module | Entrée → Sortie |
|---|---|---|---|
| 1 | `ingest` | `ingest_hf_datasets.py` | datasets HF → `data/ingested/*.jsonl` |
| 2 | `centralize` | `centralize_ingested.py` | jsonl → `wolof_ingested_corpus/…parquet` |
| 3 | `merge` | `merge_corpora.py` | corpus HF + ingéré → `wolof_unified_corpus/…parquet` |
| 4 | `stats` | `corpus_stats.py` | parquet → `data/stats/unified_corpus_stats.json` |
| 5 | `datacard` | `pipeline/datacard.py` | stats → `README.md` |
| 6 | `validate` | `pipeline/quality_gates.py` | parquet+stats → rapport gates (PASS/FAIL) |
| 7 | `publish` | `pipeline/run.py` | parquet+README → HuggingFace Hub |

Le tout est orchestré par **`pipeline/run.py`** et exposé via le **`Makefile`**.

### Format des données
Parquet à 2 colonnes : `text` (string) et `sources` (list[string]). Un même
texte présent dans plusieurs datasets garde **une seule ligne** dont la liste
`sources` est fusionnée (déduplication par `text.lower().strip()`).

---

## 3. Les Quality Gates (contrôles bloquants)

Définis dans `pipeline.yaml > quality_gates`. La publication est **refusée** si
un seul échoue.

| Gate | Vérifie | Seuil par défaut |
|---|---|---|
| `schema` | colonnes `text` (str) + `sources` (list) | — |
| `min_examples` | volume minimal | ≥ 600 000 |
| `empty_texts` | aucun texte vide | ≤ 0 |
| `duplicates` | taux de doublons exacts | ≤ 0.0 % |
| `raw_text_size` | taille texte brut UTF-8 | ≥ 100 MB |
| `wolof_pct` | part de wolof (GlotLID) | ≥ 90 % |
| `no_hf_loss` | aucun texte HF existant perdu | 0 perte |

> `no_hf_loss` télécharge le parquet **actuellement en ligne** et vérifie que
> tous ses textes sont présents dans le nouveau corpus → garantit qu'une
> publication n'écrase jamais de données existantes.

---

## 4. Mise en place (une seule fois)

### 4.1 Secrets en local
```bash
cp .env.example .env
# éditer .env et renseigner HF_TOKEN (token write sur galsenai)
```
`.env` est gitignored : il ne part jamais sur GitHub.

### 4.2 Dépendances
```bash
make setup        # = cd src && uv sync --extra dev
```

### 4.3 Secret CI (pour la publication automatique)
Sur GitHub : **Settings → Secrets and variables → Actions → New secret**
- Name : `HF_TOKEN`
- Value : le token HuggingFace (write)

---

## 5. Utilisation au quotidien

```bash
make help          # liste des commandes

# Étapes individuelles
make ingest        # [1] (re)ingestion, reprend où ça s'est arrêté
make centralize    # [2]
make merge         # [3]
make stats         # [4]
make datacard      # [5]
make validate      # [6]  ← contrôle qualité, à lancer avant tout push

# Chaîne complète sans publication
make all

# Publication (lance validate en amont, refuse si gates KO)
make publish
```

Équivalent direct sans make :
```bash
cd src
.venv/bin/python -m pipeline.run all
.venv/bin/python -m pipeline.run publish
```

---

## 6. CI/CD — automatisation locale (hooks Git)

La CI/CD **par défaut est locale**, via [`pre-commit`](https://pre-commit.com/) :
indépendante de tout cloud, gratuite, exécutée sur ta machine.

### Installation (une fois)
```bash
make hooks      # = pre-commit install + pre-commit install --hook-type pre-push
```

### Ce qui se déclenche automatiquement
| Moment | Hooks |
|---|---|
| `git commit` | trailing-whitespace, end-of-file, check-yaml, detect-private-key, check-added-large-files, `ruff` (lint + format) |
| `git push` | `pytest` (bloque le push si un test échoue) |

Exécution manuelle de tous les hooks :
```bash
cd src && uv run pre-commit run --all-files
```

### Workflows GitHub Actions (optionnels)
Les fichiers `.github/workflows/ci.yml` et `publish.yml` sont **conservés** mais
en déclenchement **manuel uniquement** (`workflow_dispatch`). Ils ne servent que
si tu utilises un compte GitHub avec Actions activé. Sinon, ignore-les : tout
passe par les hooks locaux + `make`.

---

## 7. Ajouter une nouvelle source de données

1. Ajouter une entrée dans `pipeline.yaml > ingest.datasets` :
   ```yaml
   - { repo: org/mon-dataset, cols: [wolof], split: train }
   ```
2. `make ingest` (ne retraite que les nouveaux grâce à `--skip-existing`).
3. `make all` puis `make validate`.
4. Si tout est vert → `make publish` (ou via GitHub Actions).

---

## 8. Versionnement & traçabilité

- Chaque `publish` crée un **commit sur le repo HF** (historique des versions).
- `data/ingested/_ingestion_report_consolidated.json` trace la provenance et le
  nombre d'exemples gardés/rejetés par source.
- `data/stats/quality_gates_report.json` archive le résultat des contrôles.
- La **data card** (README) est régénérée automatiquement à chaque publication.

---

## 9. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `HF_TOKEN absent` | `.env` non chargé | `cp .env.example .env` puis renseigner |
| gate `wolof_pct` = N/A | stats pas générées | lancer `make stats` avant `validate` |
| gate `no_hf_loss` en `skip` | HF injoignable / hors-ligne | vérifier réseau / token |
| `min_examples` échoue | ingestion incomplète | relancer `make ingest` |
| `ModuleNotFoundError: pipeline` | lancé hors de `src/` | exécuter depuis `src/` ou via `make` |
```
