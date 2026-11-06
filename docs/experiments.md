# Journal des expériences

Ce que les tests ont établi, pour ne pas refaire des expériences coûteuses (environ 3 min 20 par
génération sur la RTX 2000 Ada, 16 Go). Les sorties brutes sont dans `data/bg_comparison/` (non versionné).

Tous les résultats ci-dessous ont été obtenus avec **Hunyuan3D-2.1**, aujourd'hui simple référence de
recherche : sa licence ne s'applique pas dans l'UE (voir `LICENSES.md`). Les conclusions sur le
détourage et sur les images d'entrée restent valables pour un autre générateur, mais doivent être
re-vérifiées avec lui.

## Détourage (octobre 2026)

- **birefnet-general** est le meilleur sur l'illustration comme sur la photo.
- **isnet-anime** est correct sur l'illustration mais ramasse du fond sur la photo.
- **u2net** laisse un halo semi-transparent : le générateur reconstruit alors le fond et produit une plaque plate.
- **L'affinage du masque** (`refine_mask` : binarisation, fermeture, épaississement des parties fines,
  suppression des îlots) réduit les cheveux flottants.

## Matting MatAnyone2 (novembre 2026) : abandonné

Testé après birefnet sur 9 images (2 graines pour 3 d'entre elles, 1 graine pour les autres).
- L'alpha est bien plus propre (2 à 5 % de pixels semi-transparents contre plus de 40 % avec birefnet seul).
- Après affinage et génération, les meshes sont équivalents sur 7 images. Le seul gain net est sur des
  cheveux bouclés (`b4e059…`), avec en contrepartie des mèches translucides perdues sur une illustration.
- Licence non commerciale : retiré du projet.

## Paramètres de Hunyuan3D

- **La graine compte bien plus que `octree_resolution`.** En 384 et en 512, une même graine donne la même
  forme ; le 512 coûte 1,75 fois plus de temps pour un gain négligeable.
- Certaines graines produisent une découpe plate même avec un bon détourage. Le rapport épaisseur/hauteur
  du mesh ne suffit pas à la détecter.

## Défauts liés aux images d'entrée

Ces défauts apparaissent quel que soit le détourage :

1. **Accessoires fins détachés du corps perdus au détourage.** Sur `3dfab…`, la lame tenue à bout de bras
   a un alpha de 0 dans tous les masques. Le générateur peut ensuite inventer une arme ailleurs, au hasard.
2. **Sujet coupé par le cadre.** Peut produire une grosse plaque qui suit le bord de l'image (`d6d2…`),
   mais pas systématiquement (`Muzark`, buste lui aussi coupé, est correct). Le seul critère « le masque
   touche le bord » ne suffit pas à prédire l'échec.
3. **Signature d'artiste** : elle devient une excroissance sur le mesh (`1464…`).
4. **Effets semi-transparents** (magie, fumée) : transformés en nappes ou en touffes arbitraires.
5. **Petite tête** : le visage est le point faible tant qu'il occupe peu de pixels (environ 40 px dans
   l'image de 512 px vue par le générateur). Sur un buste, il est nettement plus détaillé.

## Recalage d'un mesh sur la silhouette de son image (novembre 2026)

Outils : `tools/bench/fit_silhouette.py`, `tools/bench/fit_flexicubes.py`, rasteriseur maison
`fantasy_rag/diffrender/` (détails et chiffres dans son README). Deux meshes Hunyuan3D-2.1 de
référence (`1464…`, `3dfab…`), caméra orthographique recalée sur le masque birefnet affiné. Sorties :
`data/bench/2026-11-06_fit-silhouette/` et `data/bench/2026-11-06_fit-flexicubes/`.

- **Déplacement de sommets seul** (dans le plan de l'image) : silhouette presque parfaite (IoU 0,914 →
  0,996 et 0,889 → 0,993), mais **dégradation 3D là où la correction est forte**. Sur `1464…`, la
  signature d'artiste devenue excroissance dans le dos est écrasée dans le plan au lieu de disparaître :
  cape froissée et déchiquetée de côté et de dos, cheveux plus rugueux ; 1,1 à 1,4 % de faces
  retournées. **Enseignement** : le déplacement de sommets ne convient qu'aux petites corrections ;
  retirer ou ajouter de la matière demande une représentation qui change de topologie.
- **FlexiCubes** (SDF sur grille, voxel 2,5 à 4,2 px) : la matière qui se projette hors du masque est
  retirée par l'enveloppe visuelle (découpe exacte en orthographique), puis une optimisation limitée à la
  bande de silhouette corrige le reste. IoU 0,914 → 0,984 et 0,889 → 0,967 ; les zones visibles et
  cachées hors bande ne bougent pas (Chamfer ≤ 0,26 px). La signature disparaît de la silhouette et la
  cape n'est plus froissée. **Reste un moignon** dans le dos : la partie de l'excroissance qui se
  projette derrière la cape n'est pas observable depuis une seule vue. Une heuristique de retrait
  « derrière la surface avant » troue la cape : abandonnée.
- **Réglages qui comptent** : lisser la distance au masque et faire une découpe en max lisse (sinon le
  contour en marches de pixels est extrudé en stries) ; paramétrisation lisse du changement de SDF
  (sinon terrasses de voxels) ; plafond sur l'ajout de matière (sinon bulles, perles sur les cheveux).
- **Finition** par déplacement de sommets plafonné à 3 px, avec pénalité des faces retournées : IoU
  0,992 et 0,977, aucune face retournée. La pénalité seule, sans plafond, freine trop les grandes
  corrections (IoU 0,994 → 0,970 pour 1,49 → 0,59 % de faces retournées).
- **Coût** : 2 à 3 min par mesh sur la RTX 2000 Ada (moins de 1 Gio de VRAM), 10 à 25 min sur CPU en 256.
- **Limites** : la grille lisse les détails plus fins qu'un voxel (mèches, visage) ; les parties cachées
  ne sont pas contraintes ; sur `3dfab…`, la matière manquante au-delà de 4 voxels (jambes coupées par
  le cadre, creux entre le bras et le corps) n'est pas recréée.

- **Cible brute et régions d'erreur** (2026-11-07, `data/bench/2026-11-07_fit-flexicubes-v2/`, détails
  dans `fantasy_rag/diffrender/README.md`) : cible = alpha brut de birefnet (contour sub-pixel, îlots
  < 100 px retirés) et découpe et optimisation limitées aux écarts importants et compacts. Mains de
  1464 réglées (plus de pointes ni d'excroissance), signature retirée ; **stries du visage non
  réglées** (découpe rasante d'un profil qui dépasse de 26 px : une vraie grosse erreur, pas du bruit) ;
  bosses des cheveux réduites seulement ; **main tendue de 3dfab coupée** : le mesh l'a modelée dans
  un vide que le masque affiné avait comblé. **Enseignement** : le masque brut seul ne supprime pas les
  pointes, les régions si ; mais quand le mesh a été généré depuis le masque affiné, le recaler sur le
  masque brut retire aussi ce que la fermeture avait créé, parfois une main.

## Pistes non explorées

- Détourage guidé (boîte ou points, par exemple SAM 2) pour garder les accessoires, ou birefnet en plus
  haute résolution.
- Passe « zoom » sur la tête, multi-vues, texture.
- Générateurs sous licence permissive : voir les candidats dans `LICENSES.md`.
