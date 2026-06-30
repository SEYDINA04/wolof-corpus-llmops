# État d'avancement — Session split du corpus

> **But de ce fichier** : reprendre le travail rapidement. La prochaine fois,
> lire ce fichier puis continuer à la section **« PROCHAINE ÉTAPE »**.

_Dernière mise à jour : 2026-06-29_

---

## 1. Où on en est (résumé)

- Le **corpus unifié** central fait bien **1 018 520 exemples uniques** (vérifié :
  parquet local + parquet HF identiques, 0 doublon, 0 vide). Le « 816 963 » était
  une erreur de comptage manuel, pas un vrai écart.
- **Taille brute** du corpus unifié : **443.72 MB** (octets UTF-8), 432.47 M
  caractères, 72.5 M tokens. Le parquet ne pèse que 240 MB sur disque car SNAPPY
  compresse ~2×.
- On a créé un **nouveau corpus découpé** (`wolof_split_corpus`) : chaque exemple
  est coupé en **max 2 phrases**. L'unifié 1M reste **intact**.
- Publié sur HF cette session : seulement la **data card** (README avec colonne
  `#` sur les 47 sources). Le parquet HF était déjà à jour (identique).

---

## 2. Le split (fait aujourd'hui)

**Script** : `src/split_corpus.py`
**Sortie** : `src/wolof_split_corpus/data/train-00000-of-00001.parquet`
**Relancer** : `cd src && .venv/bin/python split_corpus.py`

### Règles appliquées (décidées avec l'utilisateur)
- Phrases délimitées par `. ? ! …` **+ sauts de ligne** ; ponctuation conservée.
- Regroupement **par paires → max 2 phrases / exemple**.
- Exemples contenant un **bloc de code** (` ``` `) → **gardés intacts** (ne pas
  casser le code, ex. Code-170k).
- Décimaux (`3.14`, `2.0`) **non coupés** (split seulement si espace après la
  ponctuation).
- Nettoyage : **dédup** (texte normalisé minuscule/espaces) + suppression des
  **fragments < 10 caractères**.
- Chaque morceau **hérite des `sources`** de son exemple parent.
- Traitement **en streaming** (OOM-safe) : set de hash 64 bits + ParquetWriter.

### Résultats du split
| Métrique | Avant | Après |
|---|---:|---:|
| Exemples | 1 018 520 | **1 603 239** |
| Médiane longueur | 97 car | 116 car |
| p90 / p99 | 1282 / 2622 | 662 / 2073 |
| Max longueur | 892 808 | 27 146 |
| Texte brut UTF-8 | 443.72 MB | 426.36 MB |
| Parquet disque | 240.35 MB | 234.11 MB |

- Exemples « code » gardés intacts : **179 337**
- Doublons supprimés : **114 141**
- Fragments < 10 car retirés : **12 164**

---

## 3. PROCHAINE ÉTAPE (à décider demain)

**Question ouverte** : faut-il un **garde-fou par caractères** pour les exemples
qui restent longs malgré le split ?

Après split, il reste **52 989** exemples non-code > 400 caractères :
- **49 832** = paires de 2 phrases *longues* → conforme à « max 2 phrases ».
- **3 157** = textes **sans aucune ponctuation finale** (`.?!…`) → impossibles à
  découper avec la règle actuelle.

**Options à trancher :**
1. Garder la règle « 2 phrases » pure (ne rien faire de plus).
2. Ajouter un garde-fou : forcer une coupure au-delà de N caractères (couper sur
   l'espace le plus proche) pour traiter les 3 157 (voire les 49 832).

> Décision : ____ (à remplir demain)

**Autres TODO possibles évoqués :**
- Faut-il publier `wolof_split_corpus` sur HF (nouveau repo/branche) ou le garder
  local ?
- Régénérer les stats + quality gates sur le corpus splitté ?

---

## 4. Fichiers & chemins utiles

| Élément | Chemin |
|---|---|
| Corpus unifié (1M, intact) | `src/wolof_unified_corpus/data/train-00000-of-00001.parquet` |
| Corpus splitté (nouveau) | `src/wolof_split_corpus/data/train-00000-of-00001.parquet` |
| Script de split | `src/split_corpus.py` |
| JSONL lisible (1M, pour inspection) | `src/data/processed/corpus_1M_readable.jsonl` (544 MB, gitignoré) |
| Config pipeline | `src/pipeline.yaml` |
| Repo HF dataset | `galsenai/wolof_centalized_corpus` |

> `data/` et `*.jsonl` sont **gitignorés** → ne sont pas committés.

---

## 5. Commandes mémo

```bash
# Relancer le split
cd src && .venv/bin/python split_corpus.py

# Compter les lignes d'un corpus
cd src && .venv/bin/python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('wolof_split_corpus/data/train-00000-of-00001.parquet').metadata.num_rows)"

# Pipeline standard
make all        # ingest -> centralize -> merge -> stats -> datacard -> validate
make publish    # publie sur HF (refusé si quality gates KO)

# Régénérer le JSONL lisible depuis un parquet
#   (voir l'historique : iter_batches + json.dumps, streaming)
```
