# Conception : optimisation par l'image, par zones

Étape « optimisation par l'image » du pipeline
(générateur → nettoyage → **optimisation par l'image** → retopologie et UV → texture).
Statut : conception, rien n'est implémenté. Les chiffres de mémoire sont des estimations à mesurer au
banc d'essai.

## Contraintes (décisions de l'utilisateur, 2026-11-06)

- **Pas de GPU cloud pour l'instant** : tout doit tenir sur la RTX 2000 Ada de 16 Go.
- **3 boîtes locales**, optimisées en parallèle dans un seul processus (un seul travail GPU au sens de la
  règle `gpu-et-environnement`).
- **nvdiffrast exclu.** **Kaolin écarté** : la 0.18 charge `kaolin.non_commercial` à chaque import et
  embarque deux en-têtes propriétaires dans son binaire (audit du 2026-11-06).
- Briques retenues, sous réserve de validation dans `LICENSES.md` : **FlexiCubes autonome**
  (`nv-tlabs/FlexiCubes`, Apache-2.0, pur PyTorch), **Mitsuba 3** (BSD-3) comme rendu différentiable,
  **Large Steps** (BSD-3) pour le détail sans changement de topologie.

## Pourquoi des zones

- **Résolution adaptée** : une tête qui fait 5 % de la hauteur couvre environ 13 voxels dans une grille
  globale de 256³. Une grille locale de 128³ autour de la tête lui en donne environ dix fois plus.
- **Image recadrée par zone** : chaque boîte est optimisée contre un recadrage de l'image source, ce qui
  empêche le corps d'écraser le visage dans la perte. Cela ne crée pas de détail absent de la source.
- **Pertes spécifiques** : normales fines pour le visage, rectitude et symétrie pour une lame, silhouette
  pour le corps.

## Les 3 boîtes

| Boîte | Contenu par défaut | Remarque |
|---|---|---|
| **B1 Tête** | Visage et cheveux | Grille la plus fine ; pertes de normales prioritaires. |
| **B2 Arme et main** | L'arme et la ou les mains qui la tiennent | Une main en contact avec l'arme reste dans la même boîte que l'arme, sinon chaque boîte tire de son côté. Sans arme : les deux mains. |
| **B3 Corps** | Tout le reste, y compris la main libre | Grille la plus grossière ; surtout la silhouette. |

Les boîtes se recouvrent d'une **marge** (par exemple 10 % de leur taille) pour permettre la fusion.

## Déroulement

| Phase | Contenu | Parallèle |
|---|---|---|
| 0. Préparation | Estimation des normales et de la profondeur de l'image (MoGe-2, sous réserve d'audit), puis libération de la mémoire du modèle. | Non |
| 1. Passe globale | Caméra optimisée seule, puis forme grossière du corps entier (FlexiCubes, grille globale 192³, initialisée depuis le mesh nettoyé). Fixe le repère commun. | Non |
| 2. Découpage | Masques 2D des zones sur l'image (SAM 2 guidé, détecteurs de mains et de visage, détection de l'arme par texte : outils en cours d'audit), reportés sur le mesh par projection ; les faces cachées héritent de l'étiquette la plus proche. Contrôle visuel du découpage avant de continuer. | Non |
| 3. Raffinement local | Pour chaque boîte : grille locale initialisée depuis la SDF globale, image et caméra recadrées (même pose), tout le reste figé. Les pixels d'une boîte masqués par une autre zone sont exclus de sa perte (occlusion). | **Oui** |
| 4. Fusion | Mélange des SDF locales dans la grille globale, avec des poids qui décroissent dans les marges de recouvrement (partition de l'unité), puis **une seule extraction** du mesh, donc sans couture. | Non |
| 5. Cohérence | Quelques itérations globales courtes pour lisser les raccords. Optionnel : passe Large Steps sur la tête. | Non |

## Budget mémoire estimé (à mesurer)

| Élément | Estimation |
|---|---|
| Grille globale 192³ (paramètres et états de l'optimiseur), phases 1, 4, 5 | environ 2,7 Go |
| 3 grilles locales 128³, phase 3, simultanées | environ 3 × 0,8 Go |
| Rendus Mitsuba des recadrages (512²) | à mesurer |
| MoGe-2, phase 0 seulement | libéré avant la phase 1 |

Le pic attendu reste nettement sous 16 Go. Sur un seul GPU, les 3 boîtes se partagent la puissance de
calcul : le gain de temps vient surtout de grilles plus petites que la grille globale équivalente.

## Ce que l'étape ne corrige pas

- Les faces cachées (dos, dessous) : figées ou seulement régularisées.
- Les accessoires absents du masque de détourage (exemple : la lame de `3dfab…`, alpha à 0) : à corriger
  en amont.
- Le manque de pixels de la source : un visage de 40 px ne donnera pas de détail fiable.
- Les effets semi-transparents.

## Risques

- **Découpage faux** : une erreur de masque devient une erreur de forme. D'où le contrôle visuel.
- **Contacts mal regroupés** : main et arme séparées produisent des déformations opposées.
- **Fusion** : marges trop étroites, et des raccords apparaissent ; trop larges, et le coût augmente.
- **Licences** : MoGe-2 (données d'entraînement) et les outils de découpage sont en cours d'audit.

## Expérience proposée au banc d'essai

Sur le jeu de test complet, une graine, à partir des meshes de la référence Hunyuan3D-2.1 (recherche
uniquement) en attendant le nouveau générateur :
1. passe globale seule ;
2. passe globale + 3 boîtes ;
3. mesure du pic de mémoire et du temps par phase.

L'évaluateur compare 1 et 2 sur la silhouette, le visage et les accessoires.
