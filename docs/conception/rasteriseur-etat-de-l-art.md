# Rasteriseur maison : confrontation à l'état de l'art

Date : 2026-11-06. Objet : challenger l'approche de `fantasy_rag/diffrender` (commit `fe19ffd`) à
partir de la littérature, avant de l'utiliser pour l'optimisation par zones
(`docs/conception/optimisation-par-zones.md`).

## Notre approche

Rasterisation dure (z-buffer), barycentriques recalculées de façon différentiable, et gradients de
silhouette par **anticrénelage analytique des arêtes**, sur le principe de Laine et al. 2020
(nvdiffrast) : pour chaque paire de pixels voisins d'ids différents, on cherche l'arête qui les sépare
et on mélange les couleurs selon la fraction de croisement `t`. Le résultat est continu en `t`, donc
dérivable par rapport aux sommets.

## Les familles de méthodes

| Famille | Principe | Image produite | Gradients | Coût |
|---|---|---|---|---|
| Rasterisation douce (SoftRas, Liu et al. 2019 ; DIB-R, Chen et al. 2019) | Chaque triangle contribue à tous les pixels voisins par une fonction lisse. | **Modifiée** (floutée) | Longue portée, y compris pour la géométrie cachée ; biaisés | Proportionnel au nombre de triangles par pixel |
| Anticrénelage analytique (Laine et al. 2020) : **notre choix** | Mélange par paires de pixels selon le croisement de l'arête de silhouette. | **Modifiée** aux bords | Locaux (moins d'un pixel), approchés | Faible |
| Échantillonnage des arêtes (Li et al. 2018, redner ; Loubet et al. 2019 ; Bangaru et al. 2020) | Intégrales de bord exactes sur les discontinuités de visibilité, par lancer de rayons. | Exacte | Sans biais | Élevé, implémentation complexe |
| Micro-arêtes (Pidhorskyi et al., ECCV 2024, « Rasterized Edge Gradients ») | Le pixel est vu comme couvert ou non ; chaque paire de pixels voisins d'ids différents est une micro-arête dont le gradient est calculé au retour. | **Inchangée** | Locaux, plus justes que l'anticrénelage analytique ; gèrent les auto-intersections | Faible |
| Masquage stochastique de l'opacité (DiffSoup, 2026) | Tirage aléatoire de la visibilité, gradients par rapport de vraisemblance, plus micro-arêtes pour les positions. | Exacte par tirage | Occultation et apparition de géométrie | Faible ; vise les « soupes » de triangles |

## Ce que la littérature reproche à notre approche

Source principale : Pidhorskyi et al. 2024 (arXiv 2405.02508), qui comparent directement à nvdiffrast.

1. **Arêtes manquées.** « If triangles containing silhouette edges do not overlap with any pixel
   center, they will be overlooked during the antialiasing process. » Notre code a exactement cette
   propriété : l'arête n'est trouvée que si un triangle qui la porte couvre le centre d'un pixel. Les
   structures fines (lames, doigts, mèches) et les triangles plus petits qu'un pixel perdent leurs
   gradients de silhouette, alors que ce sont nos cibles prioritaires.
2. **Gradients approchés.** « Analytical antialiasing is inherently approximate, so the obtained
   gradients are also approximate. » Erreur relative mesurée sur leurs trois scènes : nvdiffrast 45,9 %,
   69,3 % et 39,4 % ; micro-arêtes 6,0 %, 3,4 % et 8,4 % (référence : lancer de rayons). Notre mélange
   additif des quatre voisins, approché sur les diagonales, est dans la même catégorie.
3. **Image modifiée.** L'anticrénelage change l'image produite. Pour un masque c'est voulu, mais pour
   une carte de **normales** ou de **profondeur** cela mélange deux surfaces différentes au bord, et la
   perte compare alors à la cible un signal qui n'existe pas.
4. **Auto-intersections.** Non traitées par l'anticrénelage analytique ; traitées par les micro-arêtes.

Limites reconnues par les auteurs des micro-arêtes eux-mêmes : précision moindre avec des triangles
plus petits qu'un pixel, et pas de transparence.

## Autres enseignements

- **Portée des gradients.** Toutes les méthodes par arêtes ont des gradients limités à environ un pixel.
  Pour la phase de caméra et les grands écarts, il faut procéder du grossier au fin (pyramide
  d'images), ou employer une perte de silhouette à longue portée (par exemple une carte de distance au
  bord de la cible) plutôt qu'une perte pixel à pixel.
- **Stabilité.** Nicolet et al. 2021 (« Large Steps in Inverse Rendering of Geometry ») montrent que la
  géométrie retournée ou cachée ne reçoit aucun gradient et que les erreurs s'aggravent ensuite ; leur
  préconditionnement laplacien permet de grands pas stables. C'est encore plus vrai pour nous : une
  seule vue, donc des faces cachées.
- **Taille des triangles.** Toutes les méthodes par rasterisation se dégradent quand les triangles sont
  plus petits qu'un pixel. Dans l'optimisation par zones, la résolution de grille de chaque boîte et la
  résolution de son recadrage doivent être choisies pour que les triangles fassent au moins 1 à 2 pixels
  (ou rendre en sur-échantillonnage).

## Recommandations

| Priorité | Changement | Pourquoi |
|---|---|---|
| 1 (fait) | **Ajouter les gradients par micro-arêtes** (Pidhorskyi et al. 2024) comme mode de silhouette, en réécriture d'après l'article, avec un passage avant inchangé et un retour personnalisé (`torch.autograd.Function`). Garder l'anticrénelage actuel comme second mode. | Gradients nettement plus justes, image non modifiée (normales et profondeur intactes), auto-intersections gérées. Notre code a déjà l'énumération des paires de pixels voisins. Attention : l'article contient une incohérence de signe entre l'équation 11 et l'annexe (équation 26) à trancher par un test. |
| 2 (fait) | **Test de justesse des gradients** contre une référence : couverture calculée par sur-échantillonnage fin, dérivée par différences finies, sur un triangle fin et sur un triangle plus petit qu'un pixel. DRTK (MIT) peut servir de référence externe en dehors du produit. | Les tests actuels vérifient la cohérence (gradcheck) et la convergence, pas l'erreur par rapport à la vraie dérivée de la couverture. C'est la mesure qu'utilise la littérature. |
| 3 | **Règle de dimensionnement** dans l'optimisation par zones : triangles d'au moins 1 à 2 pixels dans le recadrage, sinon sur-échantillonnage du rendu. | Contourne le défaut commun « arêtes manquées » et « triangles sous-pixel ». |
| 4 | **Perte à longue portée** pour la phase de caméra (pyramide et carte de distance de la silhouette cible). | Gradients locaux sinon. |
| 5 | Pas de rasterisation douce dans un premier temps. | Image modifiée et gradients biaisés ; seulement en secours si la phase grossière ne converge pas. |

Ces recommandations relèvent de l'agent Intégrateur ; l'Évaluateur jugera le gain sur le banc d'essai.

## Résultats de la mise en œuvre (2026-11-06)

Les priorités 1 et 2 sont faites (`fantasy_rag/diffrender/edgegrad.py`, `reference.py` ; détail et
tableaux dans `fantasy_rag/diffrender/README.md`, section « Justesse des gradients »). Ce qu'elles
changent aux recommandations :

- **Le signe est tranché** : l'équation 11, (I_A − I_B), est juste ; l'équation 26 de l'annexe
  donne le gradient opposé (test contre différences finies d'une couverture sur-échantillonnée).
- **Micro-arêtes par défaut**, y compris pour les masques : à 1 échantillon par pixel, erreur
  relative du gradient de 0,8 % (arête diagonale) à 17 % (occultation), contre 6 % à 165 % pour
  notre anticrénelage. L'ordre de grandeur rejoint celui de l'article (micro-arêtes 3 à 8 %,
  nvdiffrast 39 à 69 %). L'anticrénelage reste disponible ; son image adoucie peut servir pour
  un masque comparé à une cible douce.
- **Notre anticrénelage est plus faux que prévu sur les diagonales** : le mélange additif des
  quatre voisins compte une arête à 45° deux fois (erreur 99 %). Sa direction est bonne, ce qui
  explique que les tests de convergence passaient.
- **Deux écarts à l'article**, mesurés : (a) le report vers les sommets passe par le point de
  l'arête traversée et non par le fragment au centre du pixel, sans quoi l'épaisseur d'un
  triangle plus fin qu'un pixel n'a aucun gradient ; (b) aux intersections obliques, la dérivée
  en profondeur est pondérée par n_x² ou n_y², sans quoi le déplacement de la ligne est compté
  deux fois (mesuré : 2,0 fois à 45°). D'après notre lecture, l'équation 12 appliquée paire par
  paire a ce défaut.
- **La règle de dimensionnement (priorité 3) est confirmée et précisée** : il faut au moins deux
  échantillons sur la largeur d'une structure. Une lame de 0,45 px n'est retrouvée (épaisseur à
  0,003 px près) qu'avec un rendu à 4 × 4 échantillons par pixel ; à 1 échantillon par pixel,
  les micro-arêtes la font disparaître (image binaire face à une cible douce) et l'anticrénelage
  se trompe de 0,5 px. Un triangle qui ne couvre aucun centre d'échantillon reste invisible pour
  les deux modes. Dans l'optimisation par zones : recadrage assez fin pour que les structures
  visées fassent au moins deux pixels, sinon rendu sur-échantillonné × 4 puis moyenne.
- Coût : identique pour les deux modes sur CPU (490 ms contre 504 ms par étape, 1 million de
  faces en 512²). Mesure GPU reportée (verrou pris).

Priorités suivantes inchangées : perte à longue portée pour la phase de caméra (4), pas de
rasterisation douce (5).

## Sources

- S. Pidhorskyi et al., « Rasterized Edge Gradients: Handling Discontinuities Differentiably »,
  ECCV 2024, [arXiv 2405.02508](https://arxiv.org/abs/2405.02508). Code : DRTK (`facebookresearch/DRTK`,
  MIT d'après GitHub).
- S. Laine et al., « Modular Primitives for High-Performance Differentiable Rendering »,
  SIGGRAPH Asia 2020.
- S. Liu et al., « Soft Rasterizer », ICCV 2019, [arXiv 1901.05567](https://arxiv.org/abs/1901.05567) ;
  W. Chen et al., « DIB-R », NeurIPS 2019.
- T.-M. Li et al., « Differentiable Monte Carlo Ray Tracing through Edge Sampling », SIGGRAPH Asia
  2018 ; G. Loubet et al., « Reparameterizing Discontinuous Integrands for Differentiable Rendering »,
  2019 ; S. Bangaru et al., « Unbiased Warped-Area Sampling », 2020 (cités par Pidhorskyi et al.).
- B. Nicolet, A. Jacobson, W. Jakob, « Large Steps in Inverse Rendering of Geometry », ACM TOG 40(6),
  2021, DOI 10.1145/3478513.3480501.
- « DiffSoup: Direct Differentiable Rasterization of Triangle Soup », 2026,
  [arXiv 2603.27151](https://arxiv.org/abs/2603.27151), code `kenji-tojo/diffsoup` (MIT d'après GitHub).
