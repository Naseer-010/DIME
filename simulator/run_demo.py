from __future__ import annotations

import argparse
import time
from typing import Any

import pygame

from .backend_bridge import TASK_IDS
from .cascade_guard_env import ACTION_LABELS, C, CascadeGuardMissionControlEnv

STEP_INTERVAL = 0.42
DEFAULT_SEED = 7

ACTION_KEYS = {
    pygame.K_1: 1,
    pygame.K_2: 2,
    pygame.K_3: 3,
    pygame.K_4: 4,
    pygame.K_5: 5,
}

TRACKED = (
    pygame.K_SPACE,
    pygame.K_r,
    pygame.K_ESCAPE,
    pygame.K_1,
    pygame.K_2,
    pygame.K_3,
    pygame.K_4,
    pygame.K_5,
)


def rising(keys: Any, prev: dict[int, bool], key: int) -> bool:
    return bool(keys[key]) and not prev.get(key, False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the simulator frontend.")
    parser.add_argument("--task", choices=TASK_IDS, default="traffic_spike")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--step-interval", type=float, default=STEP_INTERVAL)
    parser.add_argument("--mute", action="store_true")
    return parser


def draw_overlay(surface: Any, env: CascadeGuardMissionControlEnv, queued: int, paused: bool) -> None:
    if not paused or env.episode_over:
        return

    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))

    card = pygame.Rect(350, 190, 900, 420)
    pygame.draw.rect(surface, C["bg_panel"], card, border_radius=22)
    pygame.draw.rect(surface, C["border"], card, 2, border_radius=22)
    pygame.draw.rect(surface, C["accent"], (card.x, card.y, card.w, 6), border_radius=6)

    font_big = pygame.font.SysFont("segoe ui", 30, bold=True)
    font_mid = pygame.font.SysFont("segoe ui", 17)
    font_mono = pygame.font.SysFont("consolas", 14)

    def write(text: str, font: Any, color: tuple[int, int, int], x: int, y: int) -> None:
        surface.blit(font.render(text, True, color), (x, y))

    write("MISSION BRIEFING", font_big, C["text"], card.x + 34, card.y + 26)
    write(env._scenario_name, font_mid, C["accent"], card.x + 36, card.y + 72)

    lines = [
        "Space: start or pause the live backend run",
        "1: reroute hottest traffic lane",
        "2: restart the hottest failed node",
        "3: scale out temporary capacity",
        "4: throttle ingress",
        "5: balance queue pressure",
        "Tab / click: focus a node",
        "R: reset the current task",
        "Esc: exit",
    ]
    y = card.y + 120
    for line in lines:
        write(line, font_mono, C["text_dim"], card.x + 36, y)
        y += 28

    hint = env.task_hint[:118] + ("..." if len(env.task_hint) > 118 else "")
    write(f"Queued action: {ACTION_LABELS.get(queued, ACTION_LABELS[0])}", font_mid, C["focus"], card.x + 36, card.y + 330)
    write("Task hint", font_mid, C["text"], card.x + 36, card.y + 352)
    write(hint, font_mono, C["text_dim"], card.x + 36, card.y + 382)


def main() -> None:
    args = build_parser().parse_args()

    env = CascadeGuardMissionControlEnv(
        render_mode="human",
        enable_audio=not args.mute,
        max_steps=900,
    )

    paused = True
    queued = 0

    def overlay(surface: Any) -> None:
        draw_overlay(surface, env, queued, paused)

    env._overlay_renderer = overlay
    env.reset(seed=args.seed, options={"task": args.task})

    running = True
    next_at = time.perf_counter() + args.step_interval
    prev = {key: False for key in TRACKED}

    while running and not env.quit_requested:
        env.render()
        if env.quit_requested:
            break

        keys = pygame.key.get_pressed()

        if rising(keys, prev, pygame.K_ESCAPE):
            running = False
        elif rising(keys, prev, pygame.K_r):
            env.reset(seed=args.seed, options={"task": args.task})
            paused = True
            queued = 0
            next_at = time.perf_counter() + args.step_interval
        elif rising(keys, prev, pygame.K_SPACE):
            paused = not paused
            next_at = time.perf_counter() + args.step_interval

        for key, action in ACTION_KEYS.items():
            if rising(keys, prev, key):
                queued = action

        now = time.perf_counter()
        if not paused and not env.episode_over and now >= next_at:
            _, _, terminated, truncated, _ = env.step(queued)
            queued = 0
            next_at = now + args.step_interval
            if terminated or truncated:
                paused = True

        for key in TRACKED:
            prev[key] = bool(keys[key])

    env.close()


if __name__ == "__main__":
    main()
