---
name: evaluation-mesh
description: Juger une campagne du banc d'essai (meshes générés à partir d'images de concept) avec des critères fixes et rendre un verdict argumenté. À utiliser par l'agent Évaluateur, qui n'a pas écrit le code évalué.
---

# Évaluation des meshes

## Principes

- **Regarder l'image source avant de conclure.** Un élément présent dans le mesh peut être réel (dans
  l'image) ou inventé ; un élément absent peut avoir été perdu dès le détourage. Vérifier le masque
  (alpha sur la zone) avant d'accuser le générateur.
- Comparer seulement à protocole identique (mêmes images, mêmes graines).
- Distinguer **différence** et **amélioration**. Si l'écart entre variantes est du même ordre que l'écart
  entre deux graines, conclure « équivalent ».

## Grille par image

Pour chaque image et chaque variante, noter `OK`, `défaut mineur` ou `échec`, avec une phrase :

1. **Silhouette** : le mesh vu de face correspond-il au sujet détouré ?
2. **Volume** : vues de côté et de dos plausibles, pas de plaque plate ni de découpe en carton.
3. **Accessoires** : armes, objets tenus, présents et au bon endroit. Vérifier d'abord qu'ils sont dans
   le masque.
4. **Tête et visage** : lisibles ; noter la taille de la tête dans l'image d'entrée.
5. **Artefacts** : morceaux flottants, excroissances (signature, texte, effets semi-transparents).

## Défauts connus à rechercher

Voir `docs/experiments.md` : accessoires fins perdus au détourage, sujet coupé par le cadre (plaque
possible), signature d'artiste, effets semi-transparents, petite tête.

## Mesures automatiques (à outiller)

À développer avec l'agent Intégrateur, puis à rapporter à chaque campagne :
- recouvrement entre la silhouette du mesh rendu de face et le masque d'entrée (IoU) ;
- similarité CLIP entre le rendu de face et l'image source ;
- nombre de composantes connexes et part des faces hors de la composante principale ;
- détection de plaque : grande face plane alignée sur un bord du cadre.

## Verdict

Un tableau image × variante, puis une conclusion en une phrase : adopter, rejeter ou « équivalent ».
Chaque conclusion cite les images qui la fondent. Ajouter les enseignements durables à
`docs/experiments.md`.
