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
- **Aucune sortie générée dans git** : rendus, planches de comparaison, meshes, masques, poids. Les écrire
  sous `data/` (campagnes : `data/bench/`). Avant un commit, vérifier `git status` et retirer tout
  fichier de ce type qui apparaîtrait.
- Toujours travailler sur une branche ; jamais de commit direct sur `master`. Une PR par sujet.
- Lancer `python -m pytest tests/ -q` avant toute PR et rapporter le résultat tel quel.
- Rapporter les résultats honnêtement : un test raté, une étape sautée ou une conclusion incertaine se
  disent explicitement.

## Quand un problème résiste

Le projet est un projet de recherche : face à un problème technique qui résiste (artefacts, méthode qui
ne converge pas, résultat contre-intuitif), **c'est à l'agent de juger** s'il a besoin de consulter la
littérature (arXiv, actes de conférences, documentation des méthodes) avant de proposer une hypothèse ou
une correction. Quand il s'appuie sur des articles :
- les citer (auteurs, titre, année, lien) dans le rapport et, si la méthode est retenue, dans la
  documentation du code ;
- distinguer ce que l'article établit de ce qui n'est qu'une transposition à notre cas, à vérifier par un
  test ;
- vérifier la licence de tout code associé avant de l'utiliser (règle `licences`) ; une idée publiée se
  réimplémente librement, un code non commercial ne se copie pas.
