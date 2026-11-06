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
    rasterize, interpolate, face_attributes, antialias, edge_gradients,
    rasterize_uv, bake_vertex_attributes, bake_face_attributes, bake_from_view,
    vertex_normals, face_normals, cube, uv_sphere,
)
```

| Fonction | Rôle | Gradients |
|---|---|---|
| `rasterize(vertices_clip, faces, resolution, max_fragments=2**22, cull_backfaces=False)` | `RasterOutput` : `triangle_id` `[B, H, W]` (−1 = fond), `bary` `[B, H, W, 3]` barycentriques corrigées de la perspective, `depth` `[B, H, W]` profondeur NDC z/w ; z-buffer. | `bary`, `depth` par rapport à `vertices_clip` (intérieur des triangles) |
| `interpolate(attributes, rast, faces)` | Attributs par sommet `[V, C]` ou `[B, V, C]` → image `[B, H, W, C]` (0 sur le fond). Couleurs, normales, positions, UV… | attributs et positions |
| `face_attributes(values, rast)` | Valeur constante par face → image. | valeurs |
| `edge_gradients(image, rast, vertices_clip, faces, scatter="edge")` | **Micro-arêtes** (Pidhorskyi et al. 2024) : renvoie l'image **inchangée** et ajoute au retour les gradients de silhouette, d'occultation et d'auto-intersection. Mode recommandé (voir « Quel mode choisir »). `scatter="fragment"` : report vers les sommets écrit comme dans l'article, pour comparaison. | image (identité) et positions, y compris la profondeur aux intersections |
| `antialias(color, rast, vertices_clip, faces)` | **Anticrénelage analytique** des arêtes de silhouette et d'occultation (principe de Laine et al. 2020) : l'image est **modifiée** aux bords (couverture fractionnaire). | couleur et positions (pas la profondeur) |
| `Camera(view, projection, resolution)` | `to_clip(v)`, `mvp`, `crop(box_px, resolution=None)` : sous-fenêtre de l'image, même pose. | |
| `perspective`, `perspective_from_intrinsics`, `look_at`, `crop_projection` | Matrices 4×4 (convention OpenGL). | différentiable si construit à partir de tenseurs |
| `rasterize_uv(uv, uv_faces, resolution)` | Rasterisation de la carte UV : face et barycentriques par texel. | |
| `bake_vertex_attributes`, `bake_face_attributes` | Attribut par sommet ou par face → texture + masque. | |
| `bake_from_view(image, vertices, faces, uv, uv_faces, mvp, texture_resolution, depth_tolerance, normals)` | Projette une image vue par une caméra sur la texture, avec test d'occultation et poids cos(vue, normale) optionnel. | par rapport à l'image |
| `vertex_normals`, `face_normals` | Normales différentiables. `cube`, `uv_sphere` : primitives de test. | |

Exemple, pertes de silhouette, de normales et de profondeur :

```python
cam = Camera(look_at((0, 0, 3)), perspective(45.0), (512, 512))
head = cam.crop((180, 40, 340, 200), resolution=(512, 512))   # boîte B1, même pose
clip = head.to_clip(vertices)                                   # vertices.requires_grad_()
rast = rasterize(clip, faces, head.resolution)
mask = rast.mask.float().unsqueeze(-1)

# Normales et profondeur : micro-arêtes, l'image n'est pas modifiée (pas de mélange de deux
# surfaces au bord) et les gradients d'occultation arrivent quand même aux sommets.
normals = edge_gradients(interpolate(vertex_normals(vertices, faces), rast, faces), rast, clip, faces)
depth = edge_gradients(rast.depth.unsqueeze(-1), rast, clip, faces)

# Masque : les deux modes sont possibles.
alpha = edge_gradients(mask, rast, clip, faces)[..., 0]          # gradients plus justes
# alpha = antialias(mask, rast, clip, faces)[..., 0]             # bords adoucis dans l'image

loss = ((alpha - target_alpha) ** 2).mean() + normal_loss(normals) + depth_loss(depth)
loss.backward()
```

Structures plus fines que deux pixels (lames, doigts, mèches) : rendre à 4 × 4 échantillons par
pixel puis moyenner, quel que soit le mode :

```python
s = 4
rast = rasterize(clip, faces, (512 * s, 512 * s))
alpha = edge_gradients(rast.mask.float().unsqueeze(-1), rast, clip, faces)
alpha = alpha.view(1, 512, s, 512, s, 1).mean(dim=(2, 4))[..., 0]
```

### Quel mode choisir

| Usage | Mode | Pourquoi (mesures ci-dessous) |
|---|---|---|
| Normales, profondeur, toute image où mélanger deux surfaces n'a pas de sens | `edge_gradients` | Image inchangée ; gradients d'occultation et d'auto-intersection. |
| Masque (silhouette) | `edge_gradients` de préférence ; `antialias` possible | Micro-arêtes : erreur de gradient de 0,8 % à 17 % sur les arêtes, occultations et intersections, contre 6 % à 165 % pour l'anticrénelage. L'anticrénelage garde un intérêt : son image est adoucie, ce qui la rapproche d'une cible douce quand on rend à 1 échantillon par pixel. |
| Structures de moins de deux pixels | l'un ou l'autre, **sur-échantillonné × 4** | À 1 échantillon par pixel, aucun mode ne retrouve l'épaisseur d'une lame de 0,45 px (tableau « Lame fine »). |
| Intersections de deux parties du mesh | `edge_gradients` | L'anticrénelage n'a aucun gradient de profondeur. |

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
4. **Micro-arêtes (silhouettes, `edge_gradients`).** Passe avant : identité. Au retour, chaque paire
   de pixels voisins A, B d'ids différents est une micro-arête placée entre les deux pixels. Son
   gradient par rapport à sa position p (en pixels, de A vers B) est
   dL/dp = ½ (dL/dI_A + dL/dI_B)(I_A − I_B) (équation 11 de l'article). La paire est classée comme
   dans l'article (section 4.3) : **fond** (un seul id valide) et **recouvrement** (un seul des deux
   centres est dans le triangle de l'autre pixel) envoient le gradient au seul triangle de premier
   plan ; **triangles adjacents** (aucun centre dans l'autre triangle) : aucun gradient ;
   **auto-intersection** (chaque centre dans le triangle de l'autre) : le gradient déplace la ligne
   d'intersection des deux triangles (section 4.4, équation 12). Trois points sont les nôtres :
   - **Signe.** L'annexe de l'article (équation 26) écrit (I_B − I_A). C'est l'équation 11 qui est
     juste : un test la compare à la référence sur-échantillonnée, et le signe opposé donne un
     gradient opposé (erreur relative 200 %).
   - **Report vers les sommets.** L'article passe par le fragment au centre du pixel (dp/dr = 1,
     puis ses barycentriques). Nous suivons le point de l'arête traversée : on calcule de façon
     différentiable le croisement t de l'arête avec le segment entre les deux centres, en figeant
     la coordonnée perpendiculaire à la paire, et l'autograd envoie dL/dt aux deux sommets de
     l'arête avec les poids (1 − s, s). Avec le fragment, les deux arêtes d'un triangle plus fin
     qu'un pixel partagent le même fragment et leurs gradients d'épaisseur s'annulent : la
     variante de l'article (`scatter="fragment"`) ne peut pas faire varier la largeur d'une lame
     (testé).
   - **Intersections obliques.** Un changement de profondeur déplace la ligne d'intersection
     selon sa normale n à l'écran ; sur un segment horizontal (vertical), le croisement bouge de
     δ/|n_x| (δ/|n_y|). Compté sur les deux axes, ce déplacement est compté deux fois pour une
     ligne oblique : mesuré à 2,0 fois à 45°, et exact seulement pour une ligne verticale ou
     horizontale. D'après notre lecture, l'équation 12 appliquée paire par paire a la même
     propriété. Nous pondérons la dérivée en profondeur d'une paire horizontale (verticale) par
     n_x² (n_y²) : la somme des deux axes redonne exactement le flux du déplacement à travers la
     ligne (erreur 0,3 % à 1,7 % sur trois orientations).

## Justesse des gradients

`fantasy_rag/diffrender/reference.py` calcule une **référence** : couverture de chaque pixel par
64 × 64 échantillons réguliers (même z-buffer, donc occultations et intersections comprises), et
dérivée de la perte par différences finies centrées (pas de 0,25 pixel). Perte linéaire
L = Σ g · I avec des poids g lisses ; image de 16 × 16 pixels, sommets en espace écran (w = 1).
La référence est stable : moins de 1 % d'écart avec 128 × 128 échantillons et un pas de
0,1 pixel (2 % pour l'occultation). Mesure : erreur relative ‖g − g_réf‖ / ‖g_réf‖ sur toutes les
coordonnées des sommets (x, y, et z pour l'intersection). Colonnes « × 4 » : rendu à 4 × 4
échantillons par pixel, puis moyenne (16 fois plus de pixels).

| Cas | Anticrénelage | Micro-arêtes | Micro-arêtes, report par fragment (article) | Anticrénelage × 4 | Micro-arêtes × 4 |
|---|---|---|---|---|---|
| Arête verticale | 5,7 % | 2,2 % | 7,4 % | 5,0 % | 1,1 % |
| Arête diagonale | **99 %** | 0,8 % | 19 % | 95 % | 0,1 % |
| Lame fine (≤ 0,6 px) | 147 % | 42 % | 101 % | 858 % | 15 % |
| Triangle sous-pixel (0,7 px, couvre un centre) | 349 % | 84 % | 102 % | 265 % | 16 % |
| Triangle sous-pixel sans centre couvert | 100 % | 100 % | 100 % | 129 % | 12 % |
| Occultation (deux profondeurs) | 148 % | 17 % | 23 % | 94 % | 6,3 % |
| Auto-intersection | 165 % | 12 % | 14 % | 100 % | 2,8 % |

Lecture :
- **L'anticrénelage surestime d'environ deux fois les arêtes diagonales** : le mélange additif des
  quatre voisins compte une arête oblique sur les deux axes. La direction du gradient reste
  bonne, ce qui explique que les optimisations convergent quand même, mais sa norme et sa
  répartition entre sommets sont fausses. Il n'a aucun gradient de profondeur aux intersections.
- **Micro-arêtes** : meilleur des modes à 1 échantillon par pixel dans tous les cas. Les erreurs
  restantes viennent de deux sources : les rangées dont aucun centre n'est couvert ne voient pas
  une structure fine (lame, sous-pixel), et dL/dI est moyenné sur les deux pixels de la paire
  (exact seulement si la perte varie peu d'un pixel à l'autre).
- **Échecs documentés** (`test_gradient_known_failures`) : triangle qui ne couvre aucun centre
  (invisible, aucun gradient dans les deux modes), lame et triangle sous-pixel à 1 échantillon par
  pixel, anticrénelage sur les diagonales, l'occultation et l'intersection. Le sur-échantillonnage
  × 4 corrige les micro-arêtes (toutes les erreurs ≤ 16 %), pas l'anticrénelage.
- Référence, cas et tableau : `python -m fantasy_rag.diffrender.reference` (environ 20 s sur CPU).

### Lame fine

`reference.blade_fit` : quadrilatère de deux triangles, 0,45 px de large, légèrement incliné,
dans une image de 32 × 32 ; cible : couverture sur-échantillonnée 64 × 64. Départ décalé de
0,73 px, largeur 1,3 px ; perte L2, Adam, 150 itérations. « Convergé » : à partir de quelle
itération les deux erreurs restent sous 0,05 px.

| Mode | Échantillons par pixel | Erreur de position | Erreur de largeur | Convergé |
|---|---|---|---|---|
| Anticrénelage | 1 | 0,012 px | 0,52 px (largeur 0,97) | non |
| Micro-arêtes | 1 | 0,031 px | 0,46 px (**largeur → 0**, la lame disparaît) | non |
| Micro-arêtes, fragment | 1 | 0,010 px | 0,06 px (bloquée dès qu'elle passe sous 1 px) | non |
| Anticrénelage | 2 | 0,007 px | 0,12 px | non |
| Micro-arêtes | 2 | 0,007 px | 0,19 px | non |
| Anticrénelage | 4 | 0,001 px | 0,047 px | itération 122 |
| **Micro-arêtes** | 4 | 0,003 px | **0,003 px** | **itération 56** |
| Micro-arêtes, fragment | 4 | 0,003 px | 0,007 px | itération 68 |

La position est retrouvée dans tous les cas. L'épaisseur demande au moins deux échantillons sur
la largeur de la structure. Avec un rendu à 1 échantillon par pixel, l'image des micro-arêtes est
binaire face à une cible douce, et un pixel couvert à 45 % pousse toujours la lame à rétrécir
jusqu'à disparaître (testé : `test_thin_blade_collapses_without_supersampling`). Même avec une
lame de 1,5 px rendue à 1 échantillon par pixel, les trois modes gardent un biais de largeur de
0,1 à 0,2 px (anticrénelage 0,12 px, micro-arêtes 0,19 px ; la variante fragment s'effondre).

## Sources

Écrit à partir d'articles et de principes généraux. **Aucun code de nvdiffrast, nvdiffrec ou
`kaolin/non_commercial` n'a été lu ni repris**, et aucun code tiers n'a été copié. Le code de
DRTK (MIT) n'a pas été consulté non plus : `edgegrad.py` est écrit à partir du texte de l'article.

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
- S. Pidhorskyi, T. Simon, G. Schwartz, H. Wen, Y. Sheikh, J. Saragih, « Rasterized Edge
  Gradients: Handling Discontinuities Differentiably », ECCV 2024,
  [arXiv 2405.02508](https://arxiv.org/abs/2405.02508) (sections 4.1 à 4.4 et annexe) :
  micro-arêtes, équation 11, classification des paires, équation 12. Réécrit d'après l'article ;
  le signe, le report vers les sommets et la pondération des intersections obliques sont
  discutés dans « Algorithme ».
- PyTorch3D (BSD-3) : non utilisé ni copié ; seulement la convention générale « triangle id +
  barycentriques par pixel », commune à tous ces rasteriseurs.

## Mesures

`tools/bench/bench_diffrender.py` (sphère UV couvrant 49 % de l'image, triangles sous-pixel, une
« étape » = rasterisation + interpolation des normales + silhouette anticrénelée + backward vers les
sommets). Meilleur temps sur 2 ou 3 répétitions, torch 2.6.0+cu124, 2026-11-06.

Comparaison des deux modes (`--mode analytic | micro_edge`), CPU, 512², même session (machine
chargée par un autre travail, d'où des temps différents du premier tableau). En mode micro-arêtes,
l'étape passe **deux** images par `edge_gradients` (masque et normales) :

| Cas | Anticrénelage | Micro-arêtes |
|---|---|---|
| 998 000 faces | 504 ms | 490 ms |
| 19 880 faces | 174 ms | 181 ms |

Le coût des deux modes est le même ordre de grandeur que la rasterisation. Mesure GPU du mode
micro-arêtes non faite : le verrou `data/.gpu-lock` était pris (expérience TRELLIS.2).

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
- **Anticrénelage : mélange additif des quatre voisins** : une arête diagonale contribue sur les
  deux axes ; la couverture est exacte pour une arête verticale ou horizontale (testé), et le
  gradient d'une arête à 45° est environ deux fois trop grand (mesuré, « Justesse des gradients »).
- **Micro-arêtes** : structures de moins de deux échantillons de large mal estimées (sur-échantillonner) ;
  dL/dI moyenné sur les deux pixels de chaque paire ; pas de transparence. Comme l'anticrénelage,
  un triangle qui ne couvre aucun centre d'échantillon n'existe pas pour lui.
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

`tests/test_flexicubes.py` (CPU, une quinzaine de secondes) : FlexiCubes creux identique à la version
dense d'origine, extraction d'une SDF analytique (erreur < 0,4 voxel, normales vers l'extérieur), SDF
d'une sphère par open3d et aller-retour, signe juste à côté d'un trou, recalage qui fait disparaître
une excroissance hors silhouette sans toucher au reste. `tests/test_fit_silhouette.py` : plafond de
déplacement, faces retournées, poids laplacien normalisé sur la sphère de test.
`tests/test_gpu_lock.py` : verrou du dépôt principal, pris, respecté, libéré.

`tests/test_diffrender.py` (CPU, une trentaine de secondes) : couverture et barycentriques d'un
triangle contre un calcul analytique, interpolation corrigée de la perspective (points sur le plan
du triangle, reprojetés sur les centres de pixels), z-buffer indépendant de l'ordre, cube vu de face,
élimination des faces arrière, paquets de fragments, lots et triangles derrière la caméra, recadrage
identique à la sous-fenêtre de l'image complète, intrinsèques, `gradcheck` en float64 de
`rasterize`, `interpolate` et `antialias`, couverture fractionnaire exacte, mini-optimisations
(translation d'une sphère et d'un cube retrouvée à partir de trois silhouettes, sommets libres d'une
sphère gonflés vers une sphère cible), cuisson UV (valeur par face aux bons texels, îlots, positions
interpolées, projection d'une image avec occultation). Micro-arêtes : passe avant identique et
gradient d'image transmis tel quel, lots, signe de l'équation 11 contre les différences finies,
équation 12 retrouvée sur une intersection, report par arête indépendant pour une lame fine,
erreur relative du gradient par cas et par mode contre la référence sur-échantillonnée (seuils et
échecs documentés), lame fine retrouvée en × 4 et effondrement en × 1 ; les mini-optimisations
existantes tournent dans les deux modes.

## Application : recalage par la silhouette

`tools/bench/fit_silhouette.py` recale un mesh existant sur le masque de son image source (phase 1
de `docs/conception/optimisation-par-zones.md`) et écrit sous `data/bench/` le mesh recalé, une
planche avant/après, la courbe de perte, `summary.json` et, avec `--blender`, les rendus des deux
meshes.

1. **Caméra seule** : orthographique (w = 1), échelle, translation 2D, petite rotation, au-dessus
   d'une orientation de face choisie parmi six (±X, ±Y, ±Z) par recouvrement après 30 itérations.
   Pyramide 128 → 256 → 512 (grand côté).
2. **Déformation, caméra figée** : déplacements des sommets **dans le plan de l'image seulement**
   (la profondeur de chaque sommet ne change pas), paramétrés par d = Sᵏ u (S : moyenne avec les
   voisins, k = 30), descente de type Adam à dénominateur uniforme (comme Large Steps, sans la
   dépendance), terme laplacien ‖L d‖² et rappel ‖d‖². Pyramide 256 × 2 → 512 × 2 (rendu
   sur-échantillonné 2 × 2).
3. **Pertes** : longue portée Σ α · φ, où φ est la distance signée au bord du masque cible (son
   minimum est atteint pour α = cible, et la force sur un bord est proportionnelle à son écart), plus
   L2 sur la silhouette ; gradients par `edge_gradients`.

Premiers résultats (2026-11-06, CPU, deux meshes Hunyuan3D-2.1 de référence, usage interne) : IoU
0,889 → 0,993 et 0,914 → 0,996 ; 0,4 à 0,8 million de faces, 5 et 14 minutes. La silhouette est
retrouvée, mais les grandes corrections (une excroissance à faire disparaître, un tissu à amincir)
se font en écrasant la surface dans le plan : 1 à 1,4 % de faces retournées et des plis visibles sous
d'autres angles. Une correction de ce type demande un changement de topologie (FlexiCubes) plutôt
qu'un déplacement de sommets.

Ajouts du 2026-11-06 (fin de journée), pour s'en servir comme **finition** après FlexiCubes :

- `--max-disp-px` : plafond doux du déplacement (|d| → c · tanh(|d| / c)), en pixels de la cible ;
- `--w-flip` / `--flip-cos` : pénalité des faces retournées, moyenne de relu(cos₀ − cos(n₀, n))² sur
  les faces (n₀ : normale initiale) ;
- terme laplacien **normalisé** par le carré de la longueur moyenne des arêtes (en hauteurs de cible) :
  le poids ne dépend plus de la densité du mesh. `w_lap = 2` remplace les anciens 2e5 (meshes de 0,4 à
  1,3 M faces) et 2e3 (sphère de test de 1 k sommets) : même IoU sur 3dfab (0,9937 contre 0,993) et
  sur la sphère ; `--lap-unnormalized` rend l'ancien terme ;
- `--device cuda|auto` prend le verrou GPU du **dépôt principal** (`tools/bench/gpu_lock.py`, aussi
  depuis une copie de travail) et reste sur CPU s'il est pris.

Mesure de la pénalité sur 3dfab, recalage complet sans plafond : faces retournées 1,49 % → 0,59 %
(w = 1) → 0,22 % (w = 10), mais IoU 0,994 → 0,970 → 0,935 : elle freine trop les grandes corrections
pour servir seule, elle convient à la finition plafonnée (0 face retournée, voir ci-dessous).

## Application : forme par FlexiCubes (changement de topologie)

`tools/bench/fit_flexicubes.py` recale la **forme** d'un mesh sur la silhouette avec FlexiCubes
(copie épinglée Apache-2.0 dans `flexicubes/`, voir son README) et le rasteriseur maison. Briques
réutilisables dans `sdf_grid.py` :

- `make_grid`, `grid_points` : grille de voxels cubiques autour de l'objet ;
- `mesh_sdf` : distance signée par open3d (`compute_distance`) ; **signe par vote de parité sur 15
  directions de rayons** (`count_intersections`). Le `compute_signed_distance` d'open3d tire ses rayons
  dans une seule direction (même avec `nsamples`) : sur une sphère dont on a ôté la calotte z > 0,9, il
  se trompe sur 8 300 points sur 195 000 (une colonne sous le trou), le vote sur 15 directions sur 0 ;
  il ne se trompe qu'à partir de très grands trous (calotte z > 0,7 : 1 % des points). Les deux meshes
  de référence sont fermés (vote unanime sur 99,9996 % des points, 0 point ambigu) ;
- `SparseGrid`, `SparseFlexiCubes` : FlexiCubes sur une **bande de cubes** seulement ; seule la levée
  d'ambiguïté C16/C19 (`_get_case_id`), qui suppose une grille dense, est réécrite (voisin trouvé par
  ses coordonnées). Testé identique à FlexiCubes dense sur un champ bruité.

Déroulé :

1. caméra : `camera.json` de `fit_silhouette` (orthographique) ;
2. SDF du mesh aux sommets de la grille (256 sur le plus grand côté par défaut sur GPU), tronquée à
   ± 6 voxels. **Non-régression** : extraite sans optimisation, elle redonne l'entrée à 0,06 px
   (1464, voxel 2,5 px) et 0,18 px (3dfab en 384, voxel 4,2 px) de distance de Chamfer ;
3. **découpe par l'enveloppe visuelle** : avec une caméra orthographique, l'enveloppe visuelle d'un
   masque est son extrusion le long de l'axe de vue, et la distance d'un point à cette enveloppe est la
   distance 2D de sa projection au masque. La SDF de l'intersection est max(s₀, d_masque / échelle) :
   une découpe exacte à toutes les profondeurs. La distance au masque est lissée (gaussienne de 1,5 px)
   et le max est un max lisse (rayon 1,5 voxel), sinon les marches de pixels du contour sont extrudées
   en stries sur la paroi découpée ;
4. **optimisation** des valeurs de SDF des seuls sommets **actifs** (projection à moins de ~11 à 14 px
   du bord du masque, |SDF| < 6 voxels) ; tous les autres sommets sont **figés** (un seul point de vue
   ne contraint pas les faces avant et arrière vues à travers la silhouette). Pertes : longue portée
   + L2 sur la silhouette (`edge_gradients`, sur-échantillonnage × 2, pyramide 256 → 512), L_dev et
   entropie croisée des signes de FlexiCubes, fidélité à la SDF de départ. Paramétrisation **lisse** du
   changement (S⁴⁸ u, S : moyenne avec les voisins de grille, sinon chaque sommet bouge seul et la
   surface se couvre de terrasses de voxels) et **plafond doux de 4 voxels sur l'ajout de matière**
   (au-delà, abaisser la SDF loin de la surface crée des bulles dans l'air : perles sur les cheveux).
   Seuls les cubes proches des sommets actifs sont ré-extraits à chaque itération ; le reste de la bande
   est rendu une fois par niveau et composé (α = 1 − (1 − α_dyn)(1 − α_fixe)) ;
5. **finition** : déplacement de sommets de `fit_silhouette`, plafonné à 3 px, pénalité des faces
   retournées w = 1 ;
6. mesures (IoU globale, tête, bords ; normales opposées à celles de la face d'entrée la plus proche ;
   Chamfer à l'entrée par zone : visible, cachée, bande de silhouette), planche, rendus Blender.

Résultats (2026-11-06, RTX 2000 Ada, deux meshes Hunyuan3D-2.1 de référence, usage interne) :

| | 1464 (voxel 2,5 px, grille 256) | 3dfab (voxel 4,2 px, grille 384) |
|---|---|---|
| IoU entrée → FlexiCubes → + finition | 0,914 → 0,984 → **0,992** | 0,889 → 0,967 → **0,977** |
| IoU tête | 0,829 → 0,956 → 0,984 | 0,915 → 0,974 → 0,984 |
| IoU bords (± 4 px) | 0,473 → 0,806 → 0,906 | 0,481 → 0,780 → 0,841 |
| Chamfer visible / caché (px) | 0,12 / 0,07 | 0,26 / 0,21 |
| Faces retournées (finition) | 0 | 0 |
| Temps (GPU) | 2 min 33 optimisation + 17 s finition | 1 min 52 + 11 s |
| Pic de VRAM | 0,77 Gio | 0,54 Gio |

Les zones cachées et visibles hors bande ne bougent pas (Chamfer ≤ 0,26 px, du même ordre que
l'extraction sans optimisation) ; toutes les corrections sont dans la bande de silhouette. Sur 1464,
la signature disparaît de la silhouette et la cape n'est plus froissée de côté ni de dos ; **reste un
moignon** de la signature dans le dos de la cape : la partie de l'excroissance qui se projette à
l'intérieur du masque (derrière la cape) n'est pas observable depuis une seule vue. Une heuristique
« retirer la matière derrière la surface avant, près d'une grosse excroissance » a été essayée et
abandonnée : elle troue la cape (une étoffe oblique n'a pas une épaisseur constante le long du rayon).
Autre limite : l'extraction à 256 ou 384 lisse les détails plus fins qu'un voxel (mèches,
visage) ; la finition ne les recrée pas.

### Cible brute et régions d'erreur (2026-11-07)

Retour de l'utilisateur sur la version du 2026-11-06 : la silhouette s'améliore mais **visage et
mains sont déformés** (stries sur la mèche et le front, bosses sur le dessus des cheveux, pointes sur
le pouce et la main au couteau). Deux corrections (la cible brute n'est plus le défaut depuis la v3, voir plus bas) :

1. **Cible = alpha brut du détoureur** (`birefnet-general_raw.png`, `--mask`), et non plus le masque
   affiné (`refine_mask`, conçu pour Hunyuan3D : il ferme les trous et épaissit les parties fines,
   +1,7 % de pixels sur 1464, +4 % sur 3dfab). Traitement de l'alpha doux : cible binaire = α > 0,5
   (mesures, régions, zones) ; α continu dans la perte L2 ; **distance au contour 0,5 sub-pixel**
   (α suréchantillonné × 4 en bilinéaire : erreur moyenne 0,08 px contre 0,36 px au pixel sur un disque
   anti-crénelé). Les îlots détachés de moins de 100 px sont retirés de la cible : l'alpha brut de 1464
   garde les traits de crayon de la signature, autour desquels l'enveloppe laissait des miettes.
   `--target refined` reprend l'ancien masque ; les IoU contre l'autre masque sont dans
   `summary.json` (`vs_refined` ou `vs_raw`).
2. **Ne corriger que les grosses erreurs** : écarts entre la silhouette de l'entrée et la cible, en
   trop et en manque séparément ; ouverture morphologique d'un disque de 3 px (élimine les liserés
   jusqu'à ~6 px de large), composantes d'au moins 300 px, puis retour aux pixels d'erreur autour des
   noyaux gardés ; poids lissé de 1 sur la région à 0 à 8 px (`regions.png`). La **découpe** n'agit que
   dans les régions en trop (ailleurs l'enveloppe est la cible plus la silhouette de l'entrée, élargie
   d'une marge, mélange lisse dans la transition) ; l'**optimisation** n'active que les sommets des
   régions, son changement est atténué par le poids et la perte de silhouette pondérée par pixel.
   `--all-errors` rend l'ancien comportement.

Résultats (`data/bench/2026-11-07_fit-flexicubes-v2/`, même caméra que le 2026-11-06 ; variantes
dans `_variants/` : `raw_all` = masque brut seul, `refined_regions` = régions seules) :

| IoU après finition (brut / affiné) | 1464 | 3dfab |
|---|---|---|
| entrée | 0,907 / 0,914 | 0,868 / 0,889 |
| 2026-11-06 (affiné, toutes erreurs) | 0,979 / **0,992** | 0,951 / **0,977** |
| **défaut** (brut + régions) | **0,984** / 0,974 | **0,955** / 0,940 |
| brut seul | 0,993 / 0,979 | 0,978 / 0,946 |
| régions seules | 0,971 / 0,980 | 0,944 / 0,968 |

Défaut : Chamfer visible / caché 0,10 / 0,07 px (1464) et 0,18 / 0,17 px (3dfab) ; normales opposées
0,66 % et 1,12 % (0,43 % et 0,75 % le 2026-11-06) ; 5 et 7 faces retournées par la finition (0
avant) ; optimisation 65 s et 56 s (au lieu de 2 min 33 et 1 min 52 : 3 fois moins de sommets actifs),
pic de VRAM 0,67 et 0,48 Gio.

Contrôle visuel (zooms 2048 et 3072 px, `tools/bench/zoom_sheet.py`, `<stem>/zooms/`) :

- **Mains de 1464 : réglé.** Main au couteau identique à l'entrée (plus d'excroissance), plus de
  pointe sous le poing. Le pouce reste raccourci : il est dans une région en trop de 407 px.
- **Signature de 1464 : retirée** de la silhouette, paroi de découpe nette (texture de voxels visible
  de trois quarts) ; le moignon caché derrière la cape reste.
- **Visage de 1464 : pas réglé.** Les stries du front et de la mèche sont dans toutes les variantes,
  dès la découpe : le profil du visage du mesh dépasse celui de l'image de 26 px d'épaisseur
  (2 455 px), une grosse erreur compacte que la découpe retire. Ce ne sont ni les marches de pixels
  (le contour sub-pixel ne change rien) ni des vides entre couches (5 % des rayons de la région en
  traversent plusieurs) : la paroi de découpe est presque parallèle à une surface elle-même tangente au
  regard au bord de la silhouette, et leur intersection alterne bandes coupées et non coupées au gré
  des ondulations des mèches. Les bosses du dessus des cheveux sont réduites mais pas supprimées : le
  masque brut manque encore 700 et 470 px au sommet du crâne, et l'ajout de matière les crée.
- **Main tendue de 3dfab : régression.** Dans l'image, du fond sépare la main (qui tient une carte) de
  l'étoffe en dessous ; le masque affiné avait comblé ce vide et Hunyuan3D a modelé la main dans ce
  pont. Contre le masque brut, la découpe retire le pont et la main avec, laissant des fragments
  flottants (même chose avec le masque brut seul ; la variante régions seules garde la main). Tête et
  poing de 3dfab : proches de l'entrée.
- **Masque brut seul** (`raw_all`) : la main au couteau de 1464 garde un rebord froissé et creusé, les
  bosses des cheveux restent : le masque brut seul ne fait pas disparaître les pointes, ce sont les
  régions qui le font.
- **Finition** : elle ne réintroduit pas de pointe visible sur les mains (une légère sur le pouce de
  1464), mais retourne maintenant 5 à 7 faces.

### Masque affiné, découpe des seuls excès détachés, finition par remaillage continu (2026-11-07, v3)

Retour sur la v2 : mains de 1464 réglées (régions d'erreur, gardées), mais **main tendue de 3dfab
coupée** par le masque brut, **stries du visage** de 1464 et **dents de scie** sur les bords d'étoffe
de 3dfab dès la découpe. Trois changements, par défaut :

1. **Cible = le masque qui a servi à générer le mesh** : pour Hunyuan3D, le masque affiné
   (`--target refined` par défaut, `--mask` reste l'alpha brut, le masque affiné est trouvé à côté) ;
   `--target raw` reste disponible.
2. **Plus de découpe générale par l'enveloppe** (`--carve detached` par défaut). Appuis : Wang et al.
   2018 (arXiv 1809.03451) montrent que l'enveloppe visuelle d'une seule vue est peu informative près des
   vues de face ; Leung et al. 2022 (CVPR workshop L3D-IVU) que le raffinement par la seule silhouette
   dégénère sans régularisation de déplacement et de lissage. Transposition (non établie par ces
   articles) : notre découpe orthographique est une enveloppe à une vue extrudée le long du regard, et
   les stries et dents de scie apparaissent là où elle coupe une surface vue de biais. Le retrait de
   matière passe donc par l'optimisation régularisée (sommets actifs aussi hors de la cible dans les
   régions en trop, quelle que soit leur profondeur dans la matière). La découpe ne reste que pour les
   **excès détachés** : au moins 300 px à plus de 0,06 × √(aire de la cible) de la cible (27 px sur
   1464, 26 px sur 3dfab) ; parmi les régions en trop des deux images, seule la signature de 1464 (39 px
   de profondeur, 497 px au-delà) l'est ; le profil du visage (25 px, 0 px au-delà) ne l'est pas. Le
   critère est calé sur ces deux images. La découpe suit la cible élargie de 3 voxels (la paroi reste
   hors du contour, l'optimisation retire le reste) avec une transition de 16 px. Ajout de matière
   plafonné à 1,5 voxel (4 avant : crête à côté de la main au couteau) ; composantes connexes de moins
   de 1 % de la plus grande retirées (bouts de mèches laissés en l'air).
3. **Finition par remaillage continu** (`--finish remesh` par défaut ; `--finish vertex` = ancienne
   finition). Palfinger 2022 (CAVW 33(5), DOI 10.1002/cav.2101) : remaillage à chaque itération
   (effondrement des arêtes courtes et des faces retournées, découpe des longues, bascules vers la
   valence 6). Code MIT copié dans `remeshing/` (commit épinglé, `torch_scatter` remplacé par
   `scatter_reduce_` et `index_add_`, voir son README). Le remaillage n'agit que dans les régions
   d'erreur. **L'optimiseur de l'article n'est pas utilisé** : essayé tel quel puis avec gradients
   projetés et lissés, il froisse la surface (pas normalisé par sommet : un petit gradient isolé déplace
   un sommet d'un pas entier ; laplacien sur la géométrie, en 3D). Pas retenu : pas dans le plan le long
   de la normale, lissé (16 moyennes avec les voisins), Adam à dénominateur uniforme, laplacien du
   champ de déplacement, plafond 8 px, pas atténué par le poids des régions.

Résultats (`data/bench/2026-11-07_fit-v3/`, même caméra, RTX 2000 Ada) :

| | 1464 (grille 256) | 3dfab (grille 384) |
|---|---|---|
| IoU affiné : entrée → découpe → FlexiCubes → remaillage | 0,914 → 0,925 → 0,956 → **0,975** | 0,889 → (pas de découpe) → 0,937 → **0,947** |
| IoU brut, après finition | 0,966 | 0,930 |
| v2 (défaut) après finition, affiné / brut | 0,974 / 0,984 | 0,940 / 0,955 |
| Chamfer visible / caché (px) | 0,11 / 0,07 | 0,21 / 0,17 |
| Normales opposées (aire) | 0,11 % | 0,28 % |
| Faces retournées par la finition (contre FlexiCubes) | 74 (0,01 % de l'aire) | 66 (0,01 %) |
| Faces : entrée → FlexiCubes → remaillage | 1 263 986 → 590 008 → 597 162 | 417 662 → 398 624 → 402 084 |
| Temps : SDF, FlexiCubes, finition | 66 s, 89 s, 19 s | 38 s, 68 s, 12 s |
| Pic de VRAM | 0,73 Gio | 0,52 Gio |

Contrôle visuel (zooms 2048 et 3072 px, lignes `input`, `v2`, `v3_*`) :

- **Visage de 1464 : stries réglées** (face et trois quarts, à toutes les étapes v3). Yeux, nez,
  bouche intacts ; le profil recule un peu. Reste de fines ondulations sur la mèche gauche du front
  après le remaillage.
- **Main tendue de 3dfab : réglée** (attachée et identique à l'entrée). Tête et poing proches de
  l'entrée ; plus de morceau flottant à gauche de la tête.
- **Dents de scie de l'étoffe de 3dfab : nettement réduites**, les marches de voxels de l'entrée
  elle-même restent.
- **Mains de 1464 : moins bien qu'en v2.** Le masque affiné diffère du mesh autour de la main au
  couteau (régions en trop et en manque) : une petite crête près du pouce apparaît dès FlexiCubes,
  adoucie par la finition ; petite bosse sous le poing (déjà en v2).
- **Signature de 1464** : retirée de la silhouette ; de trois quarts, un moignon plus petit qu'en v2
  mais strié par la paroi de découpe reste derrière la cape (non observable depuis une vue).
- **Pas réglé** : les pointes des lanières d'étoffe de 3dfab qui dépassent la cible ; deux lanières
  sous le genou sont coupées en leur milieu par l'optimisation puis retirées comme composantes isolées
  (2 588 faces) au lieu d'être déplacées ; IoU plus basses qu'en v2 (pas de découpe).

Sur CPU (4 fils), une itération coûte environ 1 s en 128, 3 à 5 s en 256 et 8 à 10 s en 384 :
compter 10 à 25 min par mesh en 256 ; sur GPU 0,2 à 1 s.

### Finition limitée au contour (2026-11-07, v3b)

Demande : la finition dégradait le détail fin (ondulations sur la mèche et la joue de 1464, crête près
du pouce et base de la lame froissée sur la main au couteau). Changement (`ContourBand`, défaut) : seuls
les sommets qui se projettent à moins de `--finish-contour-px` = **8 px** du bord de la silhouette
FlexiCubes ou de la cible bougent ou sont remaillés (poids 1 jusqu'à 4 px, smoothstep vers 0 à 8 px,
`--finish-contour-transition-px`). 8 px = plafond de déplacement de la finition et transition des
régions d'erreur : un sommet du contour peut parcourir tout le plafond et ceux juste derrière le
suivent. Les autres sont figés exactement : poids mis à 0, et `remesh()` reçoit un nouvel argument
`collapse_mask_column` qui interdit l'effondrement d'une arête touchant un sommet figé (un effondrement
déplace ses deux extrémités ; sans lui, un sommet figé pouvait bouger, voir `tests/test_remeshing.py`).
Même bande pour `--finish vertex` (`fit_silhouette.deform(vertex_weight=...)`).
`--finish-contour-px 0` rend le comportement v3. Mesure ajoutée : `frozen_max_dev_px`, distance des
sommets FlexiCubes figés à la surface finale.

Résultats (`data/bench/2026-11-07_fit-v3b/`, SDF, caméra et mesh FlexiCubes de la v3 réutilisés) :

| | 1464 v3 → v3b | 3dfab v3 → v3b |
|---|---|---|
| IoU affiné / brut après finition | 0,9752 / 0,9656 → 0,9752 / 0,9656 | 0,9465 / 0,9296 → 0,9464 / 0,9294 |
| Chamfer visible / caché (px) | 0,106 / 0,067 → 0,103 / 0,066 | 0,212 / 0,174 → 0,207 / 0,173 |
| Normales opposées (aire) | 0,108 % → 0,113 % | 0,282 % → 0,280 % |
| Faces retournées (aire) | 74 (0,012 %) → 94 (0,013 %) | 66 (0,013 %) → 69 (0,013 %) |
| Sommets figés, écart max | — → 81 %, 0,0 px | — → 80 %, 0,0 px |
| Finition, pic de VRAM | 19 s, 0,73 Gio → 22 s, 0,66 Gio | 12 s, 0,52 Gio → 15 s, 0,47 Gio |

Contrôle visuel (mêmes recadrages que la v3) : **l'intérieur est identique à FlexiCubes** (écart
0,0 px), mais **les défauts signalés ne sont pas réglés** : la mèche, la joue près du profil et la
main au couteau sont *sur* le contour, à moins de 8 px du bord, et la v3 n'y touchait déjà presque
que là (pixels modifiés par rapport à FlexiCubes sur la vue de face : 1,47 % en v3, 1,43 % en v3b, aux
mêmes endroits). Bandes plus étroites (`_variants/band4`, `band2`) : IoU 0,975 puis 0,974 sur 1464,
et c'est pire à l'œil, car le même déplacement se concentre sur moins de sommets : une marche nette
apparaît au bord de la bande sur la mèche et à la base de la lame. Main tendue de 3dfab gardée dans
toutes les variantes. **Enseignement** : ces ondulations viennent du pas de la finition sur le contour
lui-même (gradient de silhouette sur une seule vue, remaillage) et non d'un débordement vers
l'intérieur ; la bande garantit l'intérieur mais ne les corrige pas.

## Prochaines étapes

- Une extension CUDA n'est **pas nécessaire** pour l'instant : la version PyTorch fait 88 ms par
  itération à 1 million de faces en 512². À reconsidérer seulement si le banc d'essai de
  l'optimisation par zones montre que le rasteriseur domine le temps. Dans ce cas, garder ce module
  comme référence de test.
- Si nécessaire : rasterisation douce optionnelle pour les grands écarts initiaux, dilatation des
  textures cuites, découpage au plan proche.
