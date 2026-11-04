---
trigger: always_on
---

# Projet Genesis

Genesis (package `fantasy_rag`) fait deux choses :
- **Recherche** : indexer des fichiers STL (rendu Blender multi-vues + embeddings CLIP + ChromaDB) et
  chercher en langage naturel.
- **Génération** : transformer une image de concept fantasy en mesh 3D (détourage, affinage du masque,
  générateur image → 3D, nettoyage, export).

C'est un projet de **recherche à visée commerciale**, basé en **Europe** avec une clientèle européenne.
Le fine-tuning de modèles est envisagé.

## Références à lire avant de travailler

- `README.md` : installation, pipelines, structure.
- `LICENSES.md` : ce qui a le droit d'être utilisé. Voir la règle `licences`.
- `docs/experiments.md` : ce que les tests ont déjà établi. Ne pas refaire une expérience déjà conclue
  sans raison nouvelle.

## Conventions

- Langue : français pour les échanges, la documentation et les messages de commit ; le code (noms,
  commentaires) reste en anglais, comme l'existant.
- Imiter le style du code existant : densité de commentaires, nommage, gestion des dépendances optionnelles
  (`try/except ImportError` + drapeau `*_AVAILABLE`).
- `data/`, `ressources/`, `third_party/` et `.venv/` ne sont pas versionnés.
- Toujours travailler sur une branche ; jamais de commit direct sur `master`. Une PR par sujet.
- Lancer `python -m pytest tests/ -q` avant toute PR et rapporter le résultat tel quel.
- Rapporter les résultats honnêtement : un test raté, une étape sautée ou une conclusion incertaine se
  disent explicitement.
