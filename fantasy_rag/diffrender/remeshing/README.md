# Continuous remeshing (copie épinglée)

Optimisation directe d'un mesh avec remaillage adaptatif à chaque itération, de W. Palfinger,
« Continuous Remeshing for Inverse Rendering », Computer Animation and Virtual Worlds 33(5), 2022,
DOI 10.1002/cav.2101.

- **Source** : https://github.com/Profactor/continuous-remeshing
- **Commit épinglé** : `2e556421a6096701519c4f106d4af1f0588e19dd` (2022-10-28, « added getting
  started »).
- **Fichiers copiés** : `core/remesh.py` et `core/opt.py`, rien d'autre. `example.py`, `util/` et
  `paper/` (qui utilisent nvdiffrast, exclu dans `LICENSES.md`) ne sont pas copiés.
- **Licence** : `LICENSE` = `LICENSE` du commit épinglé (MIT, Copyright (c) 2022 Profactor GmbH).
- **Modifications** (signalées en tête de chaque fichier) :
  - `remesh.py` : les deux `torch_scatter.scatter_max(..., out=x)` deviennent
    `x.scatter_reduce_(0, index, src, reduce="amax", include_self=True)` (même résultat : maximum
    avec le contenu initial de `x`, testé contre une boucle) ; docstring brute dans
    `calc_face_normals` (avertissement de séquence d'échappement) ; `dim=-1` dans les `torch.cross` de
    `flip_edges` (dépréciation) ; argument optionnel `vertex_mask` de `flip_edges` (bascules
    limitées à une zone) ;
  - `opt.py` : `torch_scatter.scatter_mean(..., out=neighbor_smooth)` devient `index_add_` puis une
    division par le nombre de voisins (`bincount`, au moins 1) ; import relatif `.remesh` ; argument
    optionnels `flip_mask_column` et `collapse_mask_column` de `remesh()` (le second interdit
    l'effondrement des arêtes dont une extrémité est figée : un effondrement déplace ses deux
    sommets au milieu de l'arête).
- **Dépendances** : `torch` seulement (`torch_scatter` n'est pas installé).
- **Registre** : ligne « continuous-remeshing (Profactor) » de `LICENSES.md` (proposée OUI, PR #7, à
  faire valider par un humain). La ligne `torch_scatter` n'est plus nécessaire pour ce code.

Utilisé par la finition de `tools/bench/fit_flexicubes.py` (fonction `remesh_finish`), qui appelle
`remesh()` (effondrements, découpes et bascules d'arêtes, effondrement des faces retournées) à chaque
itération mais **pas** l'optimiseur `MeshOptimizer` de l'article : sur une seule vue, son pas
normalisé par sommet et son laplacien sur la géométrie froissaient la surface (voir
`docs/experiments.md`). `MeshOptimizer` reste disponible et testé.
