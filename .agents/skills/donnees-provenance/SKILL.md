---
name: donnees-provenance
description: Constituer un jeu de données d'entraînement ou de fine-tuning avec la provenance et la licence de chaque échantillon. À utiliser avant tout entraînement.
---

# Données traçables

## Règles

- Chaque échantillon a une provenance et une licence vérifiables. Sans elles, l'échantillon est exclu.
- Exclus d'office : licences non commerciales, contenus sans licence claire, images de test de `data/`,
  sorties d'un modèle interdit (Hunyuan3D-2.1 compris : entraîner sur ses sorties en fait un dérivé).
- Le jeu de données lui-même a une ligne dans `LICENSES.md` (section Données).

## Manifeste

Un fichier `manifest.jsonl` par jeu de données, une ligne par échantillon :

```json
{"id": "...", "source_url": "...", "auteur": "...", "licence": "CC-BY-4.0", "licence_url": "...",
 "attribution_requise": true, "date_collecte": "AAAA-MM-JJ", "fichiers": ["..."], "sha256": "..."}
```

Plus un `README.md` : origine, filtres appliqués, nombre d'échantillons par licence, attributions à
reproduire.

## Stockage

- Pas de données dans git : stockage hors dépôt (dossier non versionné ou stockage cloud), le manifeste
  et le README peuvent être versionnés.
- Garder les sommes `sha256` pour pouvoir prouver ce qui a servi à entraîner chaque modèle.

## Entraînement

- Uniquement sur un modèle de base au verdict `OUI` dans `LICENSES.md`.
- Noter pour chaque run : modèle de base et version, manifeste et version, hyperparamètres, commit git,
  machine (GPU cloud au-delà de 16 Go de VRAM).
