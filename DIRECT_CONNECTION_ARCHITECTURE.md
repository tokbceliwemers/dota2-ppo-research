# Local Shadow Fiend 1v1 Architecture

The project uses two local components:

1. **Session launcher:** the local lab runner starts Dota Tools when needed,
   launches `rl_ppo_local` on `template_map`, joins the local Radiant slot, and
   owns only its loopback Python bridge.
2. **Gameplay adapter:** the custom-game Lua addon reads observable game state,
   sends semantic actions to the local bridge, executes them, and records exact
   PPO transitions.

The launcher is not a gameplay bot API. PPO requires the gameplay adapter to
record the observation, valid action mask, sampled action, reward, terminal
flag, old log-probability, and old value from the acting policy.

The Stage 3 objective is a reproducible local Shadow Fiend-versus-Shadow Fiend
environment. It will add the opponent in observable, tested increments:
passive enemy hero, frozen scripted opponent, then frozen checkpoint
opponents. A simulator candidate is only an initializer and must pass local
evaluation before it is used for real-Dota PPO.

All interaction remains local and loopback-only. This design has no public
lobby, matchmaking, UI automation, or direct game-binary control component.
