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

## Pistes non explorées

- Détourage guidé (boîte ou points, par exemple SAM 2) pour garder les accessoires, ou birefnet en plus
  haute résolution.
- Passe « zoom » sur la tête, multi-vues, texture.
- Générateurs sous licence permissive : voir les candidats dans `LICENSES.md`.
