# Problem 1 — Flappy Bird Controller

## Requirements

```
pip install pygame numpy scipy
```

## Running the game

```bash
cd problem1
python game.py
```

The game opens a window and starts in **manual** mode. Press **M** to cycle through controller modes.

## Control modes

| Mode | What runs | How to switch |
|---|---|---|
| `manual` | You control the bird — SPACE to flap | Start here, press M to leave |
| `pid` | PID controller flies autonomously | Press M once from manual |
| `mpc` | MPC controller flies autonomously | Press M twice from manual |
| `human_in_loop` | Your flaps are blended with the PID controller | Press M three times from manual |

Press **M** repeatedly to cycle: `manual → pid → mpc → human_in_loop → manual → …`

## Keyboard shortcuts

| Key | Action |
|---|---|
| `M` | Cycle to next mode |
| `R` | Reset the game (resets score and bird position) |
| `SPACE` | Flap — only active in `manual` and `human_in_loop` modes |
| `ESC` | Quit |

## Mode details

### PID (`pid`)
A PID controller with gravity feedforward drives the bird automatically. No keyboard input is needed. The controller targets the centre of each pipe gap and scales its output based on proximity to the pipe.

### MPC (`mpc`)
A Model Predictive Controller optimises a 30-step (0.5 s) sequence of control inputs at each frame using L-BFGS-B (via `scipy`). It is more computationally expensive than PID but plans ahead. If `scipy` is not installed it falls back to random shooting.

### Human-in-the-Loop (`human_in_loop`)
Press **SPACE** to flap. Your input is blended with the PID controller's output:
- **Far from pipe** — your flap carries ~50% weight.
- **Approaching pipe** (< 300 px) — controller weight increases to 70%.
- **Pipe imminent** (< 100 px) — controller takes 90% authority.
- **Near floor or ceiling** — controller overrides completely to prevent a crash.

The blend is transparent: the bird still responds to your flaps, but erratic timing near pipes is corrected automatically.

## Switching modes mid-run

You can press **M** at any point during a run. The score resets only when the bird collides or goes out of bounds (or when you press **R**), not when you switch modes.
