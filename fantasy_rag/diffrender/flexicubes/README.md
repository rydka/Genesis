# FlexiCubes (copie épinglée)

Extraction de surface différentiable de Shen et al., « Flexible Isosurface Extraction for
Gradient-Based Mesh Optimization » (SIGGRAPH 2023), code de NVIDIA.

- **Source** : https://github.com/nv-tlabs/FlexiCubes
- **Commit épinglé** : `4cc7d6c3d0cee83c011ce36721b81adff0dd7db6` (2025-04-04, « update kaolin
  functions »). La licence est **Apache-2.0 depuis `eac497031e5138aa544065f357d093f9f3e101e9`**
  (2025-04-03, « update licence ») ; avant ce commit, c'était la « NVIDIA Source Code License for
  FlexiCubes », non commerciale. Ne jamais reprendre un fichier d'un commit antérieur.
- **Fichiers copiés** : `flexicubes.py` et `tables.py`, rien d'autre. Le dossier `examples/` du dépôt
  (qui installe nvdiffrast et Kaolin, exclus dans `LICENSES.md`) n'est pas copié.
- **Licence** : `LICENSE` = `LICENSE.txt` du commit épinglé (Apache-2.0). En-têtes d'origine conservés.
- **Modification** : dans `flexicubes.py`, `from tables import *` devient `from .tables import *`
  (import relatif dans le paquet) ; la modification est signalée dans l'en-tête du fichier, comme le
  demande l'article 4(b) de la licence Apache-2.0. `tables.py` est identique à l'original.
- **Dépendances** : `torch` seulement.
- **Registre** : ligne « FlexiCubes autonome » de `LICENSES.md` (proposée OUI, PR #7, à faire valider
  par un humain).

L'adaptation à une grille creuse (bande autour de la surface) vit hors de ces fichiers, dans
`fantasy_rag/diffrender/sdf_grid.py`, par héritage.
