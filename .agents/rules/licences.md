---
trigger: always_on
---

# Politique de licences

Le projet vise un usage commercial dans l'Union européenne. Une licence incompatible peut bloquer la
commercialisation. Cette règle prime sur toute autre considération (qualité, vitesse, facilité).

1. **Rien n'entre sans ligne dans `LICENSES.md`** : bibliothèque, modèle, poids, jeu de données, outil
   téléchargé par un script. Si la ligne n'existe pas, s'arrêter et passer par `/gardien-licences`.
2. **Interdits** :
   - les licences qui excluent l'UE (exemple : Hunyuan3D-2.1) ;
   - les licences non commerciales (« non-commercial », « research only », CC-BY-NC, S-Lab…) ;
   - les dérivés de ces modèles : fine-tuning, distillation, entraînement sur leurs sorties.
3. **Hunyuan3D-2.1** reste dans `third_party/` comme **référence de recherche** pour comparer. Ne pas
   l'étendre, ne pas le fine-tuner, ne pas s'en servir pour générer des données d'entraînement.
4. **Vérifier ce qui est réellement téléchargé.** Un modèle MIT dont le script officiel télécharge un
   détoureur non commercial n'est pas utilisable tel quel. Lire le fichier LICENSE et la fiche des poids,
   pas seulement l'étiquette HuggingFace.
5. **Images de test** (`data/`) : œuvres d'artistes tiers, évaluation interne uniquement. Jamais pour
   l'entraînement, une démo publique ou du marketing.
6. Un agent peut proposer une ligne du registre ; seul un humain la valide (colonne « Validé par »).
