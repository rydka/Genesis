---
trigger: always_on
---

# GPU et environnement local

## Machine

- Windows 11, terminal **Windows PowerShell 5.1** : pas de `&&` (utiliser `;` ou `if ($?) { ... }`),
  et Python écrit sur stderr ce que PowerShell affiche à tort comme `NativeCommandError`.
- Python du projet : `.venv\Scripts\python.exe` (3.12).
- GPU unique : **NVIDIA RTX 2000 Ada, 16 Go**.
- Blender 4.5 LTS : `C:\Program Files\Blender Foundation\Blender 4.5\blender.exe` (chemin attendu par
  `fantasy_rag/config/settings.py`). Rendu de GLB : `tools/debug/render_glb_views.py`.
- Pas de `make` ni de `winget`. `gh` est dans `C:\Users\Shadow\bin`.
- Toute action qui demande les droits administrateur (installation MSI, PATH système) est faite par un humain.

## Un seul travail GPU à la fois

Deux agents qui chargent des modèles en même temps saturent les 16 Go et ralentissent tout, ou plantent.

1. Avant un travail GPU, vérifier `data/.gpu-lock`. S'il existe et que le processus qu'il cite tourne
   encore, attendre ou travailler sur autre chose.
2. Sinon, écrire dans `data/.gpu-lock` : nom de l'agent, PID, heure de début, tâche.
3. Supprimer le fichier à la fin, y compris en cas d'échec.

## Mémoire GPU

- Une session onnxruntime birefnet occupe environ 7 Go : la libérer (`BackgroundRemover.unload()`) avant
  de charger le générateur.
- Le chargement à froid du générateur actuel prend environ 4 min ; une génération environ 3 min 20.
  Regrouper les générations dans un seul processus qui charge le modèle une fois.
- Les modèles qui demandent plus de 16 Go (Direct3D-S2 en 1024, fine-tuning) passent par un GPU cloud.

## Longues tâches

Lancer les tâches de plus de quelques minutes en processus détaché avec des fichiers de log, pour
qu'elles survivent à une fin de session ; le script doit sauter les sorties déjà produites, pour pouvoir
être relancé.
