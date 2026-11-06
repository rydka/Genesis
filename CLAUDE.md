# Genesis — instructions pour Claude Code

@AGENTS.md

## Règles toujours actives

@.agents/rules/projet.md
@.agents/rules/licences.md
@.agents/rules/gpu-et-environnement.md

## Sous-agents

Les six rôles sont définis comme sous-agents dans `.claude/agents/` : `gardien-licences`,
`veille-modeles`, `integrateur`, `banc-essai`, `evaluateur`, `donnees-finetuning`. Chacun suit son
workflow de `.agents/workflows/` et le skill associé de `.agents/skills/`.

L'agent principal orchestre : un sous-agent ne peut pas en appeler un autre. Quand un workflow dit
« transmets à `/x` » ou « passe la main à `/x` », le sous-agent le signale dans son rapport et c'est
l'agent principal qui lance le sous-agent suivant, ou qui demande une décision à l'utilisateur.
