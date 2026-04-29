"""
Problem 1: Dynamical System and Control — Flappy Bird

Coordinate system: y=0 at the bottom of the screen, increasing upward.
Rendering converts to pygame screen coordinates via: screen_y = WINDOW_HEIGHT - y.

Controls (keyboard):
  SPACE   — flap (manual mode) / human flap (human-in-loop mode)
  M       — cycle through modes: manual → pid → mpc → human_in_loop
  R       — reset game
  ESC     — quit
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pygame

try:
    import numpy as np
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_WIDTH: int = 800
WINDOW_HEIGHT: int = 500
FPS: int = 60
GRAVITY: float = -80.0  # vertical acceleration (world coords, downward)
FLAP_FORCE: float = 100.0  # upward velocity given per manual flap (world coords)
BG_COLOR = (135, 206, 235)

MODES = ["manual", "pid", "mpc", "human_in_loop"]


# ---------------------------------------------------------------------------
# Bird
# ---------------------------------------------------------------------------
@dataclass
class Bird:
    x: float = 100.0
    y: float = WINDOW_HEIGHT / 2  # world y (0 = bottom)
    vx: float = 100.0  # horizontal speed (pixels/s); increases with score
    vy: float = 0.0  # vertical velocity (world coords; positive = up)
    w: float = 20.0
    h: float = 20.0


def bird_motion(bird: Bird, control: float, dt: float) -> None:
    """Update bird position and velocity.

    Args:
        bird:    Bird state (modified in place).
        control: Vertical acceleration input from the controller (world coords).
        dt:      Time step in seconds.
    """
    bird.vy += (GRAVITY + control) * dt
    bird.y += bird.vy * dt
    # Clamp to floor only; ceiling is handled by the loose out-of-bounds check.
    bird.y = max(0.0, bird.y)


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------
@dataclass
class Pipe:
    x: float = float(WINDOW_WIDTH)
    h: float = 150.0  # height of the bottom pipe section (world y from bottom)
    gap: float = 120.0  # vertical gap between bottom and top pipe sections
    w: float = 60.0


def pipe_motion(pipe: Pipe, bird: Bird, dt: float) -> bool:
    """Move pipe leftward relative to the bird and reset when off-screen.

    Args:
        pipe:  Pipe state (modified in place).
        bird:  Bird state (provides horizontal speed).
        dt:    Time step in seconds.

    Returns:
        True when the pipe resets (bird successfully passed a gap).
    """
    pipe.x -= bird.vx * dt

    if pipe.x + pipe.w < 0:
        pipe.x = float(WINDOW_WIDTH)
        pipe.h = random.uniform(60.0, WINDOW_HEIGHT - pipe.gap - 60.0)
        return True  # bird cleared a pipe

    return False


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------
def check_collision(bird: Bird, pipe: Pipe) -> bool:
    """Return True if the bird overlaps with any part of the pipe."""
    bx1, bx2 = bird.x, bird.x + bird.w
    by1, by2 = bird.y, bird.y + bird.h

    px1, px2 = pipe.x, pipe.x + pipe.w

    if bx2 <= px1 or bx1 >= px2:
        return False  # no horizontal overlap

    gap_bottom = pipe.h
    gap_top = pipe.h + pipe.gap

    # Collision if bird is below the gap bottom or above the gap top
    return by1 < gap_bottom or by2 > gap_top


# ---------------------------------------------------------------------------
# 1.1  PID Controller
# ---------------------------------------------------------------------------
@dataclass
class PIDController:
    Kp: float = 9.0   # tuned for double-integrator plant with gravity feedforward
    Ki: float = 0.1
    Kd: float = 6.0
    error_accumulator: float = 0.0
    prev_error: float = 0.0
    max_accumulator: float = 200.0
    dt: float = 1.0 / 60.0

    def reset(self) -> None:
        """Reset the controller state."""
        self.error_accumulator = 0.0
        self.prev_error = 0.0

    def calc_input(
        self,
        set_point: float,
        process_var: float,
        velocity: float = 0.0,
        umin: float = -500.0,
        umax: float = 500.0,
    ) -> float:
        """Calculate the PID control signal.

        Args:
            set_point:   Target value (desired height).
            process_var: Current measured value (current height).
            velocity:    Current vertical velocity (for derivative/feedforward).
            umin:        Minimum control output.
            umax:        Maximum control output.

        Returns:
            Clamped control signal in [umin, umax].
        """
        error = set_point - process_var

        # 1. Integral with anti-windup
        self.error_accumulator += error * self.dt
        self.error_accumulator = max(
            -self.max_accumulator, min(self.max_accumulator, self.error_accumulator)
        )

        # 2. Derivative via velocity feedback — avoids derivative kick on setpoint changes
        derivative = -velocity

        # 3. Gravity feedforward cancels the constant -80 downward acceleration so
        #    the P/I/D terms only need to handle tracking deviations.
        gravity_ff = -GRAVITY  # = 80.0

        u = (
            self.Kp * error
            + self.Ki * self.error_accumulator
            + self.Kd * derivative
            + gravity_ff
        )

        # 4. Clamp
        u = max(umin, min(umax, u))

        # 5. Store for next call
        self.prev_error = error
        return u


# ---------------------------------------------------------------------------
# 1.2  MPC Controller
# ---------------------------------------------------------------------------
class MPCController:
    """Model Predictive Controller for the Flappy Bird game.

    Optimizes a sequence of control inputs over a finite prediction horizon
    by simulating the bird dynamics and minimising a cost function.
    """

    def __init__(
        self,
        horizon: int = 30,
        dt: float = 1.0 / 60.0,
        umin: float = -500.0,
        umax: float = 500.0,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.umin = umin
        self.umax = umax
        # Warm-start: shift the previous solution by one step each call.
        self._last_inputs: list[float] = [0.0] * horizon

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _simulate(
        self,
        y0: float,
        vy0: float,
        inputs: list[float],
    ) -> list[tuple[float, float]]:
        """Simulate bird vertical dynamics over the prediction horizon.

        Args:
            y0:     Initial y position (world coords).
            vy0:    Initial vertical velocity.
            inputs: Control inputs, one per time step.

        Returns:
            List of (y, vy) for each step.
        """
        states: list[tuple[float, float]] = []
        y, vy = y0, vy0
        for u in inputs:
            vy += (GRAVITY + u) * self.dt
            y += vy * self.dt
            y = max(0.0, min(float(WINDOW_HEIGHT), y))
            states.append((y, vy))
        return states

    def _cost(
        self,
        states: list[tuple[float, float]],
        target: float,
        inputs: list[float],
    ) -> float:
        """Evaluate the MPC cost for a candidate input sequence.

        Args:
            states: Simulated (y, vy) pairs over the horizon.
            target: Desired y position (centre of the pipe gap).
            inputs: Corresponding control inputs.

        Returns:
            Scalar cost (lower is better).
        """
        n = len(states)
        cost = 0.0

        for i, (y, _) in enumerate(states):
            # Tracking error — weight increases toward end of horizon so the
            # optimizer cares more about where the bird ends up than the path.
            step_weight = (i + 1) / n
            cost += step_weight * (y - target) ** 2

            # Soft floor / ceiling barriers prevent the optimizer from planning
            # trajectories that graze the window edges.
            cost += 1e4 * max(0.0, 15.0 - y) ** 2
            cost += 1e4 * max(0.0, y - (WINDOW_HEIGHT - 15.0)) ** 2

        # Small control-effort penalty → prefer gentle inputs over bang-bang.
        cost += 1e-4 * sum(u ** 2 for u in inputs)

        # Terminal velocity penalty: arrive at target with near-zero vertical speed.
        if states:
            _, final_vy = states[-1]
            cost += 0.5 * final_vy ** 2

        return cost

    def _optimize(self, y0: float, vy0: float, target: float) -> list[float]:
        """Find the input sequence that minimises self._cost() via L-BFGS-B.

        Falls back to random shooting when scipy is unavailable.
        """
        if _SCIPY_AVAILABLE:
            def objective(u_arr: "np.ndarray") -> float:
                states = self._simulate(y0, vy0, u_arr.tolist())
                return self._cost(states, target, u_arr.tolist())

            x0 = np.array(self._last_inputs)
            bounds = [(self.umin, self.umax)] * self.horizon
            result = _scipy_minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 50, "ftol": 1e-4},
            )
            return result.x.tolist()

        # Fallback: random shooting — sample candidates, keep the best.
        best_inputs = self._last_inputs[:]
        best_cost = float("inf")
        for _ in range(200):
            inputs = [random.uniform(self.umin, self.umax) for _ in range(self.horizon)]
            states = self._simulate(y0, vy0, inputs)
            c = self._cost(states, target, inputs)
            if c < best_cost:
                best_cost = c
                best_inputs = inputs
        return best_inputs

    # ------------------------------------------------------------------
    # Public interface (same signature as PIDController.calc_input)
    # ------------------------------------------------------------------
    def calc_input(
        self,
        set_point: float,
        process_var: float,
        velocity: float = 0.0,
        umin: float = -500.0,
        umax: float = 500.0,
    ) -> float:
        """Return the first control action from the optimised sequence.

        Args:
            set_point:   Target y position (gap centre).
            process_var: Current y position.
            velocity:    Current vertical velocity (vy).
            umin:        Minimum control value (passed to optimiser).
            umax:        Maximum control value (passed to optimiser).

        Returns:
            First element of the optimised input sequence.
        """
        # Warm-start: shift previous solution left and pad with zero.
        self._last_inputs = self._last_inputs[1:] + [0.0]

        best_inputs = self._optimize(process_var, velocity, set_point)
        self._last_inputs = best_inputs

        return best_inputs[0]


# ---------------------------------------------------------------------------
# Control signal (used by both automated controllers)
# ---------------------------------------------------------------------------
def calculate_control_signal(bird: Bird, pipe: Pipe, controller) -> float:
    """Calculate the control signal for the bird.

    Args:
        bird:       Current bird state.
        pipe:       Current pipe state.
        controller: Controller instance with a ``calc_input`` method.

    Returns:
        Control signal value.
    """
    # Only consider pipes that are ahead of the bird.
    if pipe.x + pipe.w < bird.x:
        return 0.0

    # Target: centre of the gap (world y coords).
    target_height = pipe.h + pipe.gap / 2

    # Anticipate overshoot by adjusting target with current velocity.
    velocity_offset = bird.vy * 0.2
    adjusted_target = target_height - velocity_offset

    current_height = bird.y + bird.h / 2
    distance_to_pipe = pipe.x - (bird.x + bird.w)

    # Scale control aggressiveness by proximity to the pipe.
    if distance_to_pipe <= -1:
        distance_factor = 1.5
    else:
        distance_factor = max(0.5, min(1.5, 1 + 1 / (distance_to_pipe + 1)))

    return (
        controller.calc_input(adjusted_target, current_height, bird.vy)
        * distance_factor
    )


# ---------------------------------------------------------------------------
# 1.3  Human-in-the-Loop control signal
# ---------------------------------------------------------------------------
def calculate_control_signal_human(
    bird: Bird,
    pipe: Pipe,
    controller,
    human_flap: bool,
    alpha: float = 0.5,
) -> float:
    """Blend the human's flap input with an automated controller's output.

    Args:
        bird:        Current bird state.
        pipe:        Current pipe state.
        controller:  Automated controller (PID or MPC).
        human_flap:  True if the human pressed the flap key this frame.
        alpha:       Weight for the human input in [0, 1].
                     alpha=1 → pure human, alpha=0 → pure controller.

    Returns:
        Blended control signal.
    """
    human_signal = FLAP_FORCE if human_flap else 0.0
    auto_signal = calculate_control_signal(bird, pipe, controller)

    # Distance-based alpha: trust the controller more as the pipe approaches.
    distance_to_pipe = max(0.0, pipe.x - (bird.x + bird.w))
    if distance_to_pipe < 100:
        dynamic_alpha = 0.1   # pipe imminent — mostly controller
    elif distance_to_pipe < 300:
        dynamic_alpha = 0.3   # approaching — mixed
    else:
        dynamic_alpha = alpha  # far away — respect human weight

    # Safety override: near floor or ceiling the controller takes over entirely.
    near_floor = bird.y < 40.0
    near_ceiling = bird.y > WINDOW_HEIGHT - 60.0
    if near_floor or near_ceiling:
        return auto_signal

    return dynamic_alpha * human_signal + (1.0 - dynamic_alpha) * auto_signal


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _world_to_screen_y(world_y: float) -> int:
    """Convert world y (0 = bottom) to pygame screen y (0 = top)."""
    return int(WINDOW_HEIGHT - world_y)


def draw_bird(surface: pygame.Surface, bird: Bird) -> None:
    screen_y = _world_to_screen_y(bird.y + bird.h)
    rect = pygame.Rect(int(bird.x), screen_y, int(bird.w), int(bird.h))
    pygame.draw.rect(surface, (0, 200, 0), rect)


def draw_pipe(surface: pygame.Surface, pipe: Pipe) -> None:
    # Bottom pipe: from y=0 (screen bottom) up to pipe.h (world)
    bottom_h = int(pipe.h)
    bottom_rect = pygame.Rect(
        int(pipe.x), WINDOW_HEIGHT - bottom_h, int(pipe.w), bottom_h
    )
    # Top pipe: from world y = pipe.h + pipe.gap to top of window
    top_world_start = pipe.h + pipe.gap
    top_h = int(WINDOW_HEIGHT - top_world_start)
    top_rect = pygame.Rect(int(pipe.x), 0, int(pipe.w), max(0, top_h))
    pygame.draw.rect(surface, (0, 150, 0), bottom_rect)
    pygame.draw.rect(surface, (0, 150, 0), top_rect)


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flappy Bird — CS702 Asg 2")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)

    mode_idx = 0  # start in manual mode; cycle with M key
    mode = MODES[mode_idx]

    pid = PIDController()
    mpc = MPCController()

    def reset() -> tuple[Bird, Pipe, int]:
        return Bird(), Pipe(), 0

    bird, pipe, score = reset()
    human_flap = False
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        human_flap = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    bird, pipe, score = reset()
                    pid.reset()
                elif event.key == pygame.K_m:
                    mode_idx = (mode_idx + 1) % len(MODES)
                    mode = MODES[mode_idx]
                    print(f"[mode] {mode}")
                elif event.key == pygame.K_SPACE:
                    if mode == "manual":
                        bird.vy = (
                            max(0.0, bird.vy) + FLAP_FORCE
                        )  # cancel downward momentum, then flap
                    elif mode == "human_in_loop":
                        human_flap = True

        # --- Compute control ---
        if mode == "manual":
            control = 0.0
        elif mode == "pid":
            control = calculate_control_signal(bird, pipe, pid)
        elif mode == "mpc":
            control = calculate_control_signal(bird, pipe, mpc)
        elif mode == "human_in_loop":
            control = calculate_control_signal_human(bird, pipe, pid, human_flap)
        else:
            control = 0.0

        # --- Update dynamics ---
        bird_motion(bird, control, dt)
        if pipe_motion(pipe, bird, dt):
            score += 1
            bird.vx += 5.0  # speed boost

        # --- Check game-over ---
        out_of_bounds = bird.y <= 0 or bird.y > WINDOW_HEIGHT * 1.5
        if check_collision(bird, pipe) or out_of_bounds:
            print(f"[game over] score={score}  mode={mode}")
            bird, pipe, score = reset()
            pid.reset()

        # --- Draw ---
        screen.fill(BG_COLOR)
        draw_pipe(screen, pipe)
        draw_bird(screen, bird)
        hud = font.render(
            f"Score: {score}    Mode: {mode}    (M=cycle  R=reset  SPACE=flap  ESC=quit)",
            True,
            (0, 0, 0),
        )
        screen.blit(hud, (10, 10))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
