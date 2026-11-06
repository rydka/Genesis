# Instructions pour les agents (Codex et autres)

Genesis transforme des images de concept fantasy en meshes 3D et indexe des STL pour la recherche en
langage naturel. C'est un projet de **recherche à visée commerciale**, basé en **Europe**.

La configuration des agents vit dans `.agents/` (format Google Antigravity). Ce fichier la rend
utilisable par Codex sans la dupliquer.

## À lire avant toute tâche

Les règles de `.agents/rules/` s'appliquent toujours, quel que soit l'outil :

- `.agents/rules/projet.md` : contexte, conventions, git, aucune sortie générée dans le dépôt.
- `.agents/rules/licences.md` : rien n'entre sans ligne dans `LICENSES.md` ; Hunyuan3D-2.1 et toute
  licence non commerciale ou excluant l'UE sont interdits pour le produit.
- `.agents/rules/gpu-et-environnement.md` : Windows, PowerShell 5.1, un seul travail GPU à la fois
  (verrou `data/.gpu-lock`), chemins de Blender et du Python du projet.

Références : `LICENSES.md`, `docs/experiments.md`, `README.md`.

## Rôles

Le travail est réparti en six rôles. Chaque rôle est décrit dans un fichier de `.agents/workflows/`.
Quand la demande nomme un rôle (par exemple « Gardien des licences : audit complet »), lis le fichier
correspondant et suis-le étape par étape, y compris ce qu'il interdit.

| Rôle | Fichier | Ne fait jamais |
|---|---|---|
| Gardien des licences | `.agents/workflows/gardien-licences.md` | Valider une ligne de `LICENSES.md` à la place d'un humain |
| Veille modèles | `.agents/workflows/veille-modeles.md` | Installer ou intégrer |
| Intégrateur | `.agents/workflows/integrateur.md` | Merger sa propre PR |
| Banc d'essai | `.agents/workflows/banc-essai.md` | Juger les résultats |
| Évaluateur | `.agents/workflows/evaluateur.md` | Modifier le code évalué |
| Données et fine-tuning | `.agents/workflows/donnees-finetuning.md` | Entraîner sur un modèle ou des données interdits |

Sans rôle nommé, applique les règles ci-dessus et choisis le rôle le plus proche de la demande.

## Skills

Les skills de `.agents/skills/` (audit de licence, intégration de modèle, banc d'essai, évaluation de
mesh, provenance des données) sont détectés automatiquement par Codex et par Antigravity.

## Vérifications

- Tests : `.venv\Scripts\python.exe -m pytest tests/ -q` (environ 1 à 7 minutes selon la charge du GPU).
- Avant un commit : `git status`, et aucun rendu, mesh, masque ou poids dans la liste.
- Une branche par sujet, PR vers `master`, jamais de merge par l'agent lui-même.
