# Questions de licence à soumettre à un juriste

Date : 6 novembre 2026. Projet : Genesis.

## Contexte

Genesis est un projet de recherche à visée commerciale, porté par une structure établie dans l'Union
européenne, pour une clientèle européenne. Il transforme une image de concept (illustration d'un
personnage de fantasy : guerrier, magicienne, chevalier, avec épées, lances ou armes imaginaires) en
modèle 3D, destiné à des jeux vidéo ou à l'impression de figurines. Le mode de distribution n'est pas
encore arrêté : logiciel livré au client ou service en ligne.

Pour cela, le projet s'appuie sur des modèles d'intelligence artificielle publiés par des tiers (code
et « poids », c'est-à-dire les paramètres appris du modèle). Chaque composant est inscrit dans un
registre (`LICENSES.md`), avec sa licence et la phrase qui fonde notre lecture. Ce registre est une
analyse technique, pas un avis juridique : les points ci-dessous sont ceux que nous ne pouvons pas
trancher seuls.

Nos règles internes, à valider elles aussi (question 6) : aucun composant sous licence non commerciale
ou excluant l'UE ; pas de composant GPL ou AGPL chargé dans notre programme ; LGPL et MPL acceptées
sans modification.

## Questions

### 1. Licence DINOv3 (Meta) : clause sur les armes

Composant indispensable du générateur 3D que nous envisageons (TRELLIS.2, lui-même sous licence MIT).
La licence DINOv3 est commerciale et mondiale. Elle contient la clause suivante (§1.b.v) :

> « You agree not to use, or permit others to use, DINO Materials for any activities subject to the
> International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including
> those related to military or warfare purposes, nuclear industries or applications, espionage, or the
> development or use of guns or illegal weapons. »

Les « Trade Controls » incluent les sanctions de l'UE. Le contractant pour l'Espace économique européen
est Meta Platforms Ireland ; le droit applicable est celui de la Californie.

**Questions** : la création de modèles 3D de personnages imaginaires portant des armes (épées, lances,
armes fantastiques), pour le jeu ou la figurine, peut-elle relever de « the development or use of guns
or illegal weapons » ? Le choix du droit californien pose-t-il un problème particulier pour une
structure européenne ?

### 2. Licence SAM 3 (Meta) : même famille, plus une modification unilatérale

Alternative à un outil de découpage d'image que nous n'utilisons pas pour l'instant (nous avons retenu
SAM 2, sous Apache-2.0). La « SAM License » contient la même clause §1.b.v que DINOv3, et :

> §8 : « Meta may modify this Agreement from time to time […] All such changes will be effective
> immediately ».

**Question** : une licence modifiable unilatéralement et immédiatement est-elle acceptable pour un
composant d'un produit commercial, et quelles précautions prendre (archivage de la version acceptée,
etc.) ?

### 3. Pixal3D (Tencent ARC) : historique de licence

Générateur 3D candidat de repli. Son historique public montre :
- 12 mai 2026 : première licence, « academic purposes only » et « Pixal3D IS NOT INTENDED FOR USE
  WITHIN THE EUROPEAN UNION » ;
- 20 mai 2026 : remplacée par la licence MIT, sur le dépôt de code puis sur celui des poids ;
- la fiche du modèle sur la plateforme HuggingFace garde un paramètre `extra_gated_eu_disallowed: true`
  (accès bloqué depuis l'UE par la plateforme).

**Questions** : peut-on se fonder sur la licence MIT en vigueur, en utilisant une version publiée après
le 20 mai 2026 ? Le paramètre de blocage UE, contradictoire avec la licence, a-t-il une portée
juridique ? Faut-il demander une confirmation écrite à Tencent ARC ?

### 4. Poids sous licence permissive, entraînés sur des données à conditions restrictives

Plusieurs modèles publient leurs poids sous licence permissive (MIT ou Apache-2.0), alors qu'une partie
de leurs données d'entraînement est soumise à des conditions non commerciales ou propres :
- **MoGe-2** (estimation de profondeur et de normales, Microsoft, poids MIT) : ARKitScenes, Waymo,
  ScanNet++ et d'autres jeux de données dont plusieurs ont des conditions non commerciales ;
- **SAM 2** (découpage d'image, Meta, Apache-2.0) : jeu de données SA-1B, à conditions propres ;
- **Grounding DINO** (non retenu pour l'instant) : comprend Flickr30k Entities.

**Question** : les conditions attachées aux données d'entraînement peuvent-elles restreindre notre
utilisation commerciale des poids publiés, ou de ce que nous produisons avec eux ?

### 5. Réécriture « indépendante » d'un code sous licence non commerciale

Une bibliothèque nécessaire au générateur (cubvh, utilisée par CuMesh) contenait du code repris d'un
projet NVIDIA sous licence non commerciale (« may be used or intended for use non-commercially »).
Son mainteneur a publié le 26 août 2026 une réécriture présentée comme indépendante, préparée avec un
agent d'intelligence artificielle.

**Questions** : quels éléments faudrait-il pour considérer cette réécriture comme libre des droits
NVIDIA ? Le fait qu'elle ait été produite avec un outil d'IA change-t-il l'analyse ? Si le risque est
jugé trop élevé, nous réécrirons ce composant nous-mêmes.

### 6. Notre politique sur les licences « copyleft »

Nous appliquons les règles suivantes :
- GPL et AGPL : jamais chargées dans notre programme ; seulement des logiciels externes lancés à part
  (exemple : Blender, GPL-3.0, utilisé comme outil de rendu) ;
- LGPL : acceptée telle quelle ;
- MPL-2.0 : acceptée sans modification des fichiers concernés.

**Question** : ces règles suffisent-elles, à la fois pour un logiciel livré au client et pour un
service en ligne (cas de l'AGPL notamment) ? Un rasteriseur que nous écrivons nous-mêmes à partir
d'articles scientifiques, sans reprendre de code, pose-t-il une difficulté au regard des licences des
logiciels décrits dans ces articles ?

### 7. Images de test

Pour évaluer le pipeline, nous utilisons en interne neuf illustrations d'artistes trouvées en ligne
(certaines signées). Elles ne servent ni à entraîner un modèle, ni à une démonstration publique, ni au
marketing.

**Question** : cet usage interne d'évaluation est-il couvert, par exemple par l'exception de fouille de
textes et de données (directive (UE) 2019/790, article 4), ou faut-il le remplacer par des images dont
nous détenons les droits ?

## Documents joints

- `LICENSES.md` : registre complet, avec pour chaque composant la licence, la source lue et la date.
- Les textes de licence cités sont disponibles sur les dépôts publics indiqués dans le registre ; nous
  pouvons en fournir des copies datées.
