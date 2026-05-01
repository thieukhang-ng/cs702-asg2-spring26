#!/usr/bin/env python3
"""Play trajectory JSON with pygame (800x600 canvas)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pygame


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python animate.py <trajectories.json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    canvas = data.get("canvas", {})
    W = int(canvas.get("width", 800))
    H = int(canvas.get("height", 600))
    fps = float(data.get("fps", 30))
    raw = data["trajectories"]
    N = len(raw)
    K = len(raw[0])
    colors = [tuple(data["colors"][i]) for i in range(N)]
    hs = data.get("hotspots", [])

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Trajectory animation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monaco", 14)

    t_frame = 0
    running = True
    paused = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    t_frame = 0

        if not paused:
            t_frame = (t_frame + 1) % (K * 3)
        t = min(t_frame % K, K - 1)

        screen.fill((18, 18, 22))
        for h in hs:
            c = (200, 80, 80) if h["kind"] == "converge" else (80, 140, 220)
            pygame.draw.circle(screen, c, (int(h["x"]), int(h["y"])), 10, width=2)

        for i in range(N):
            pts = [(int(raw[i][s][0]), int(raw[i][s][1])) for s in range(K)]
            if len(pts) > 1:
                pygame.draw.lines(screen, colors[i], False, pts, 1)
            x, y = pts[t]
            pygame.draw.circle(screen, colors[i], (x, y), 6)

        hud = font.render("SPACE pause | R restart | time %d/%d" % (t, K - 1), True, (220, 220, 220))
        screen.blit(hud, (8, 8))
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
