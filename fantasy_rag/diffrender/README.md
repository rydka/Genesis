# `fantasy_rag.diffrender` : rasteriseur différentiable maison

Rasteriseur de triangles différentiable, en **pur PyTorch** (aucune dépendance au-delà de `torch`),
écrit pour Genesis à la place de nvdiffrast (licence non commerciale), de Kaolin (charge du code non
commercial) et de Mitsuba 3 (décision de l'utilisateur du 2026-11-06). Il sert à :

1. l'**optimisation par l'image** (`docs/conception/optimisation-par-zones.md`) : rendre un mesh vu par
   une caméra, éventuellement recadrée, et propager vers les sommets les gradients de pertes de
   silhouette, de profondeur et de normales ;
2. la **cuisson de textures** : rasteriser dans l'espace UV et projeter une image ou des attributs sur la
   texture.

Le dossier `fantasy_rag/rendering/` (rendu Blender de la recherche) n'a rien à voir avec ce module.

## API

```python
from fantasy_rag.diffrender import (
    Camera, look_at, perspective, perspective_from_intrinsics, crop_projection, transform_points,
    rasterize, interpolate, face_attributes, antialias,
    rasterize_uv, bake_vertex_attributes, bake_face_attributes, bake_from_view,
    vertex_normals, face_normals, cube, uv_sphere,
)
```

| Fonction | Rôle | Gradients |
|---|---|---|
| `rasterize(vertices_clip, faces, resolution, max_fragments=2**22, cull_backfaces=False)` | `RasterOutput` : `triangle_id` `[B, H, W]` (−1 = fond), `bary` `[B, H, W, 3]` barycentriques corrigées de la perspective, `depth` `[B, H, W]` profondeur NDC z/w ; z-buffer. | `bary`, `depth` par rapport à `vertices_clip` (intérieur des triangles) |
| `interpolate(attributes, rast, faces)` | Attributs par sommet `[V, C]` ou `[B, V, C]` → image `[B, H, W, C]` (0 sur le fond). Couleurs, normales, positions, UV… | attributs et positions |
| `face_attributes(values, rast)` | Valeur constante par face → image. | valeurs |
| `antialias(color, rast, vertices_clip, faces)` | Anticrénelage analytique des arêtes de silhouette et d'occultation. **C'est lui qui donne les gradients de silhouette.** | couleur et positions |
| `Camera(view, projection, resolution)` | `to_clip(v)`, `mvp`, `crop(box_px, resolution=None)` : sous-fenêtre de l'image, même pose. | |
| `perspective`, `perspective_from_intrinsics`, `look_at`, `crop_projection` | Matrices 4×4 (convention OpenGL). | différentiable si construit à partir de tenseurs |
| `rasterize_uv(uv, uv_faces, resolution)` | Rasterisation de la carte UV : face et barycentriques par texel. | |
| `bake_vertex_attributes`, `bake_face_attributes` | Attribut par sommet ou par face → texture + masque. | |
| `bake_from_view(image, vertices, faces, uv, uv_faces, mvp, texture_resolution, depth_tolerance, normals)` | Projette une image vue par une caméra sur la texture, avec test d'occultation et poids cos(vue, normale) optionnel. | par rapport à l'image |
| `vertex_normals`, `face_normals` | Normales différentiables. `cube`, `uv_sphere` : primitives de test. | |

Exemple, perte de silhouette :

```python
cam = Camera(look_at((0, 0, 3)), perspective(45.0), (512, 512))
head = cam.crop((180, 40, 340, 200), resolution=(512, 512))   # boîte B1, même pose
clip = head.to_clip(vertices)                                   # vertices.requires_grad_()
rast = rasterize(clip, faces, head.resolution)
alpha = antialias(rast.mask.float().unsqueeze(-1), rast, clip, faces)[..., 0]
normals = interpolate(vertex_normals(vertices, faces), rast, faces)
loss = ((alpha - target_alpha) ** 2).mean()
loss.backward()
```

Faces UV à coutures : passer les coins comme sommets (`attributes [F*3, C]`,
`faces = arange(F*3).view(F, 3)`) ; `rasterize_uv` accepte des `uv_faces` différentes des `faces`
géométriques (même nombre de faces).

## Conventions

- Espace caméra : la caméra regarde vers −Z, +Y en haut (OpenGL). Espace de découpage (`clip`) :
  NDC = xyz/w dans [−1, 1]³, y NDC vers le haut. Face avant = sens trigonométrique en NDC.
- Images : **ligne 0 en haut**, comme les tableaux numpy/PIL. Le centre du pixel (i, j) est en
  x = 2(j + 0,5)/W − 1, y = 1 − 2(i + 0,5)/H. Une image cible se compare donc sans retournement.
- Intrinsèques : origine au coin haut-gauche, centre du pixel (i, j) en (j + 0,5, i + 0,5).
- Recadrage : `box_px = (x0, y0, x1, y1)` en pixels de l'image complète ; la projection est
  composée à gauche par une transformation affine de x et y (z et w inchangés), donc la profondeur
  et le z-buffer sont identiques à ceux de l'image complète (vérifié par un test).
- Textures (OBJ/OpenGL) : u vers la droite, v vers le haut ; texel (i, j) centré en
  u = (j + 0,5)/W, v = 1 − (i + 0,5)/H. Pour du glTF (v vers le bas), passer `1 − v`.

## Algorithme

1. **Visibilité, sans gradient.** Chaque triangle est projeté, sa boîte englobante en pixels est
   énumérée en « fragments » (paires pixel × triangle), testés par fonctions d'arête. Le test de
   profondeur est un seul `scatter_reduce(amin)` sur une clé 64 bits `(bits de la profondeur float32
   ≥ 0) << 32 | id du triangle` : les motifs binaires des flottants positifs sont ordonnés comme les
   flottants, donc le minimum de la clé donne le triangle le plus proche, avec départage déterministe
   par le plus petit id. Les triangles sont traités par paquets de `max_fragments` fragments au plus :
   la mémoire ne dépend pas du nombre de faces mais de ce budget (environ 100 octets par fragment).
2. **Réévaluation différentiable.** Pour chaque pixel couvert, les barycentriques écran sont
   recalculées à partir des sommets du triangle gagnant, puis corrigées de la perspective
   (λᵢ = (bᵢ/wᵢ) / Σ bⱼ/wⱼ), et la profondeur z/w est interpolée linéairement en espace écran.
   L'autograd de PyTorch donne alors les dérivées à l'intérieur des triangles, sans backward écrit à la
   main.
3. **Anticrénelage (silhouettes).** Pour chaque paire de pixels voisins (horizontale ou verticale)
   d'ids différents : choix du triangle dont l'arête les sépare (le seul triangle de premier plan, ou
   l'occultant, déterminé par inclusion des centres, puis par profondeur), intersection de cette arête
   avec le segment entre les centres à la fraction t, puis mélange : si t > ½ le pixel voisin reçoit
   (t − ½) de la couleur du triangle, sinon le pixel du triangle reçoit (½ − t) de la couleur voisine.
   Le résultat est continu en t, donc en positions des sommets. Les arêtes partagées par les deux
   triangles de la paire (arêtes intérieures du mesh) sont ignorées.

## Sources

Écrit à partir d'articles et de principes généraux. **Aucun code de nvdiffrast, nvdiffrec ou
`kaolin/non_commercial` n'a été lu ni repris**, et aucun code tiers n'a été copié.

- J. Pineda, « A Parallel Algorithm for Polygon Rasterization », SIGGRAPH 1988 : fonctions d'arête.
- P. Heckbert et H. Moreton, « Interpolation for Polygon Texture Mapping and Shading », 1991 :
  interpolation corrigée de la perspective.
- S. Laine, J. Hellsten, T. Karras, Y. Seol, J. Lehtinen, T. Aila, « Modular Primitives for
  High-Performance Differentiable Rendering », SIGGRAPH Asia 2020 : découpage rasterisation /
  interpolation / anticrénelage et **principe** de l'anticrénelage analytique par arêtes de silhouette
  (réimplémenté d'après la description de l'article ; les règles de choix de l'arête et de mélange
  ci-dessus sont les nôtres).
- S. Liu et al., « Soft Rasterizer », ICCV 2019, et W. Chen et al., « DIB-R », NeurIPS 2019 :
  rasterisation douce, étudiée comme alternative et non retenue (voir Limites).
- PyTorch3D (BSD-3) : non utilisé ni copié ; seulement la convention générale « triangle id +
  barycentriques par pixel », commune à tous ces rasteriseurs.

## Mesures

`tools/bench/bench_diffrender.py` (sphère UV couvrant 49 % de l'image, triangles sous-pixel, une
« étape » = rasterisation + interpolation des normales + silhouette anticrénelée + backward vers les
sommets). Meilleur temps sur 2 ou 3 répétitions, torch 2.6.0+cu124, 2026-11-06.

| Cas | Appareil | Visibilité seule | Étape complète | Mémoire de pointe |
|---|---|---|---|---|
| 998 000 faces, 512² | CPU | 163 ms | 752 ms | 0,82 Gio (processus entier) |
| 998 000 faces, 512² | RTX 2000 Ada | 22 ms | 88 ms | 0,30 Gio (`max_memory_allocated`) |
| 19 880 faces, 512² | CPU | 83 ms | 277 ms | 0,60 Gio (processus entier) |
| 19 880 faces, 512² | RTX 2000 Ada | 9 ms | 57 ms | 0,12 Gio |
| 998 000 faces, 1024² | CPU | 368 ms | 1 456 ms | 1,26 Gio (processus entier) |
| 998 000 faces, 1024² | RTX 2000 Ada | 34 ms | 236 ms | 0,63 Gio |

Le cas visé (environ 1 million de faces en 512² sur 16 Go) tient avec une large marge : trois boîtes
recadrées en 512² font environ 1 Go et 0,3 s par itération sur le GPU.

## Limites connues

- **Pas de découpage au plan proche** : un triangle dont un sommet a w ≤ 0 (derrière la caméra) est
  ignoré. Sans effet pour un objet entièrement devant la caméra, ce qui est notre cas.
- **Gradients de silhouette locaux** : comme toute méthode par arêtes, ils n'existent qu'à moins d'un
  pixel des silhouettes. Il faut que les silhouettes rendue et cible se recouvrent ; pour de grands
  écarts, procéder du grossier au fin (basse résolution d'abord) ou ajouter un flou à la perte. Une
  rasterisation douce (SoftRas) donnerait des gradients à plus longue portée, au prix d'un coût
  proportionnel au nombre de faces par pixel ; non implémentée.
- **Mélange additif des quatre voisins** : une arête diagonale contribue sur les deux axes ; la
  couverture est exacte pour une arête verticale ou horizontale (testé), approchée sinon.
- **Pas de règle de remplissage haut-gauche** : un pixel dont le centre tombe exactement sur une arête
  commune est attribué au triangle le plus proche puis au plus petit id. Aucun trou, mais les ids de
  ces pixels peuvent changer avec l'arrondi.
- **Arêtes intérieures** : une arête est ignorée seulement si le triangle voisin dans la paire la
  partage ; une arête intérieure dont le voisin direct est plus petit qu'un pixel peut produire un
  petit gradient parasite.
- **Cuisson** : pas de dilatation des îlots UV (prévoir une marge de quelques texels au moment du
  dépliage, ou dilater après coup) ; les îlots UV qui se recouvrent sont attribués au plus petit id.
- Le z-buffer quantifie la profondeur sur 23 bits de mantisse (float32) ; les surfaces à moins de
  1e-7 en profondeur NDC l'une de l'autre sont départagées par id.

## Tests

`tests/test_diffrender.py` (CPU, une quinzaine de secondes) : couverture et barycentriques d'un
triangle contre un calcul analytique, interpolation corrigée de la perspective (points sur le plan
du triangle, reprojetés sur les centres de pixels), z-buffer indépendant de l'ordre, cube vu de face,
élimination des faces arrière, paquets de fragments, lots et triangles derrière la caméra, recadrage
identique à la sous-fenêtre de l'image complète, intrinsèques, `gradcheck` en float64 de
`rasterize`, `interpolate` et `antialias`, couverture fractionnaire exacte, mini-optimisations
(translation d'une sphère et d'un cube retrouvée à partir de trois silhouettes, sommets libres d'une
sphère gonflés vers une sphère cible), cuisson UV (valeur par face aux bons texels, îlots, positions
interpolées, projection d'une image avec occultation).

## Prochaines étapes

- Une extension CUDA n'est **pas nécessaire** pour l'instant : la version PyTorch fait 88 ms par
  itération à 1 million de faces en 512². À reconsidérer seulement si le banc d'essai de
  l'optimisation par zones montre que le rasteriseur domine le temps. Dans ce cas, garder ce module
  comme référence de test.
- Si nécessaire : rasterisation douce optionnelle pour les grands écarts initiaux, dilatation des
  textures cuites, découpage au plan proche.
