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

## Prochaines étapes

- Une extension CUDA n'est **pas nécessaire** pour l'instant : la version PyTorch fait 88 ms par
  itération à 1 million de faces en 512². À reconsidérer seulement si le banc d'essai de
  l'optimisation par zones montre que le rasteriseur domine le temps. Dans ce cas, garder ce module
  comme référence de test.
- Si nécessaire : rasterisation douce optionnelle pour les grands écarts initiaux, dilatation des
  textures cuites, découpage au plan proche.
