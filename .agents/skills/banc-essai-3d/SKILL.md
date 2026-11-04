---
name: banc-essai-3d
description: Lancer une campagne de comparaison de génération 3D (variantes de détourage, générateurs, graines) avec un protocole fixe, des rendus Blender et des sorties traçables. À utiliser pour toute expérience GPU sur le pipeline de génération.
---

# Banc d'essai de génération 3D

## Protocole fixe

Un résultat n'est comparable que si le protocole est identique :

- **Images** : le jeu de test `data/*.jpg|*.webp` (évaluation interne uniquement). Toujours toutes les
  images du jeu, pas une sélection.
- **Graines** : la même liste pour toutes les variantes (par défaut 1 et 2 ; une seule si le budget
  l'impose, mais la même pour toutes). La graine influence plus le résultat que la plupart des réglages :
  ne jamais comparer deux variantes sur des graines différentes.
- **Une seule variable** change entre deux variantes. Tout le reste (étapes, guidance, résolution,
  affinage du masque) est noté dans le résumé.

## Organisation des sorties

```
data/bench/<AAAA-MM-JJ>_<sujet>/
  protocol.json          # images, graines, variantes, paramètres, commit git
  <image>/<variante>_seed<N>.glb
  <image>/renders/       # 4 vues Blender par mesh
  <image>/ab_views.png   # planche : lignes = variante × graine, colonnes = vues
  summary.json           # une entrée par génération : succès, temps, sommets, faces, erreur
```

## Outils existants

- Détourage comparé : `tools/debug/compare_bg_removal.py IMAGE --models ...` (planche des masques et de
  ce que voit le générateur).
- Balayage d'hyperparamètres : `tools/debug/sweep_hunyuan.py` (spécifique à Hunyuan3D, à généraliser).
- Rendu : `"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" -b -P tools/debug/render_glb_views.py -- OUT_DIR a.glb b.glb`.

## Exécution

1. Prendre le verrou GPU (règle `gpu-et-environnement`).
2. Un seul processus charge le modèle une fois et enchaîne les générations ; il saute les `.glb` déjà
   présents pour pouvoir reprendre après une coupure.
3. Lancer en processus détaché avec logs dans le dossier du run.
4. Rendre chaque image dès que ses meshes sont prêts, puis libérer le verrou.

## Livrable

Le dossier du run et un court rapport : ce qui a tourné, ce qui a échoué (avec l'erreur), durée totale.
Le jugement des résultats revient à l'agent Évaluateur, pas au banc d'essai.
