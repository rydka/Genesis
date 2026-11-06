# Agents Genesis (Google Antigravity)

Configuration des agents qui font avancer Genesis dans Antigravity. Un « agent » est une conversation
lancée depuis l'Agent Manager avec un workflow comme point d'entrée (`/nom`). Les règles s'appliquent à
tous ; les skills sont chargés selon la tâche.

## Les six agents

| Agent | Lancer avec | Rôle | Ne fait jamais |
|---|---|---|---|
| Gardien des licences | `/gardien-licences` | Audite les composants, tient `LICENSES.md` | Valider une ligne à la place d'un humain |
| Veille modèles | `/veille-modeles` | Repère et trie les modèles candidats | Installer ou intégrer |
| Intégrateur | `/integrateur` | Branche les modèles validés, outille le banc d'essai | Merger sa propre PR |
| Banc d'essai | `/banc-essai` | Lance les campagnes GPU selon un protocole fixe | Juger les résultats |
| Évaluateur | `/evaluateur` | Juge les campagnes et rend un verdict | Modifier le code évalué |
| Données et fine-tuning | `/donnees-finetuning` | Jeux de données traçables, fine-tuning | Entraîner sur un modèle ou des données interdits |

## Enchaînement

```
Veille modèles ──> Gardien des licences ──> Intégrateur ──> Banc d'essai ──> Évaluateur ──> décision humaine
                          ^                                     ^                │
                          │                                     └──── Données et fine-tuning
                          └────────── tout nouveau modèle, dépendance ou jeu de données
```

Points de décision humains : validation des lignes de `LICENSES.md`, merge des PR, campagnes de plus de
2 heures, dépenses de GPU cloud.

## Contenu

- `rules/` (toujours actives) : `projet`, `licences`, `gpu-et-environnement`.
- `skills/` : `audit-licence`, `integration-modele`, `banc-essai-3d`, `evaluation-mesh`,
  `donnees-provenance`.
- `workflows/` : un fichier par agent.

Références communes hors de ce dossier : `LICENSES.md`, `docs/experiments.md`, `README.md`.

## Compatibilité

**Codex** lit `AGENTS.md` à la racine du dépôt et détecte les skills de `.agents/skills/`. Il ne lit pas
`rules/` ni `workflows/` de lui-même : `AGENTS.md` lui dit de lire les règles et, pour jouer un rôle,
le fichier de workflow correspondant (« Gardien des licences : … »). Règles et workflows restent donc
la source unique pour les deux outils.

La documentation d'Antigravity utilise `.agents/` ; d'anciens guides et certains outils utilisent
`.agent/`. Si les règles ou les workflows n'apparaissent pas dans Antigravity, vérifier le nom de dossier
attendu par la version installée. Le format des règles (`trigger: always_on`) et des skills
(`SKILL.md` avec `name` et `description`) suit la documentation de novembre 2026.

## Premières missions

1. `/gardien-licences` : audit complet du registre.
2. `/integrateur` : outiller les mesures automatiques du skill `evaluation-mesh` et un script de campagne
   générique dans `tools/bench/`.
3. `/veille-modeles` puis `/gardien-licences` : retenir les remplaçants de Hunyuan3D-2.1 (TRELLIS.2,
   TripoSG, Direct3D-S2, Step1X-3D).
4. `/integrateur`, `/banc-essai`, `/evaluateur` : comparer les candidats retenus sur tout le jeu de test.
