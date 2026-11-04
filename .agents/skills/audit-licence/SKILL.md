---
name: audit-licence
description: Vérifier la licence d'un composant (bibliothèque, modèle, poids, jeu de données) pour un usage commercial dans l'UE et mettre à jour LICENSES.md. À utiliser avant toute intégration ou téléchargement de modèle.
---

# Audit de licence

## Ce qu'il faut établir

Pour chaque composant, quatre choses distinctes, chacune avec sa source :

1. **Licence du code** : fichier `LICENSE` / `LICENSE.txt` / `NOTICE` du dépôt, à la version utilisée.
2. **Licence des poids** : fiche du modèle (HuggingFace `cardData.license`, fichier LICENSE du dépôt de
   poids). Elle peut différer de celle du code.
3. **Territoire et restrictions** : exclusions géographiques, plafond d'utilisateurs ou de chiffre
   d'affaires, politique d'usage acceptable, obligations d'attribution.
4. **Dépendances téléchargées** : ce que le code charge à l'exécution (autres poids, détoureur,
   encodeur). Chercher `from_pretrained`, `hf_hub_download`, `torch.hub`, URLs `http` dans le code.
   Chaque dépendance a sa propre ligne d'audit.

Pour un jeu de données : licence globale **et** licence par élément (Objaverse et similaires ont une
licence par objet), conditions sur les modèles entraînés dessus.

## Méthode

- Lire les textes, ne pas se fier aux étiquettes ni aux sites tiers (certains listent Hunyuan3D-2.1 en
  MIT, ce que son fichier LICENSE contredit).
- Liste rapide des licences déclarées sur HuggingFace :
  `Invoke-RestMethod "https://huggingface.co/api/models/<id>"` puis lire `cardData.license`.
- Citer la phrase exacte qui fonde la conclusion, surtout en cas de refus.

## Verdict

- `OUI` : MIT, Apache-2.0, BSD et équivalents, sans restriction territoriale, dépendances comprises.
- `NON` : non commercial, recherche seulement, exclusion de l'UE, ou dépendance indispensable dans ce cas.
- `À VÉRIFIER` : texte ambigu, licence maison, plafonds, dépendance non auditée.

Mettre à jour `LICENSES.md` (statut, colonnes, date, source). Ne jamais remplir « Validé par » : c'est
la décision d'un humain. En cas de `NON` sur un composant déjà utilisé, le signaler en tête du rapport.
