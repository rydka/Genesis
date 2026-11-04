---
name: integration-modele
description: Intégrer un modèle tiers (générateur image vers 3D, détoureur) dans le pipeline fantasy_rag derrière les interfaces existantes, en maîtrisant la VRAM et les dépendances. À utiliser une fois la licence validée dans LICENSES.md.
---

# Intégrer un modèle tiers

## Préalable

La ligne du composant dans `LICENSES.md` doit exister avec un verdict `OUI`, ou `À VÉRIFIER` pour une
expérience isolée sur une branche. Sinon, s'arrêter.

## Où et comment

- Cloner le dépôt dans `third_party/<Nom>` (non versionné) et documenter la commande dans le README.
- **Éviter `pip install` du paquet** quand il tire des dépendances lourdes ou des versions figées
  (interfaces graphiques, `huggingface_hub==x`) : l'ajouter à `sys.path`, comme
  `fantasy_rag/generation/shape_generator.py` le fait pour `hy3dshape`, et déclarer explicitement dans
  `config/requirements/generation.txt` les dépendances réellement importées.
- Import optionnel : `try/except ImportError` + drapeau `<NOM>_AVAILABLE` + `logger.warning`.

## Interfaces

- Générateur image → 3D : même contrat que `ShapeGenerator` (`ShapeConfig`, `generate(image_path)` qui
  renvoie un `ShapeResult`, `export_mesh`, `unload`). Si plusieurs générateurs coexistent, ajouter un
  choix de backend dans la config plutôt que de dupliquer le pipeline.
- Détourage : passer par `BackgroundRemover` (`fantasy_rag/generation/background.py`) ; garder notre
  détourage même si le dépôt tiers fournit le sien (souvent sous licence non commerciale).
- Toute nouvelle option : valeur par défaut qui conserve le comportement actuel, exposée dans
  l'interface Streamlit seulement si elle est validée par le banc d'essai.

## VRAM

- Charger paresseusement, libérer dans `unload()` (`gc.collect()` + `torch.cuda.empty_cache()`).
- Libérer le détoureur avant de charger le générateur.
- Mesurer et noter dans la PR : pic de VRAM (`torch.cuda.max_memory_allocated`), temps de chargement,
  temps par génération.

## Avant la PR

- Un test minimal sans GPU (import, configuration, comportement quand le modèle est absent).
- `python -m pytest tests/ -q` vert.
- README et `generation.txt` à jour.
