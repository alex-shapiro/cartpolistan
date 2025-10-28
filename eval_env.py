import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.classic_control import CartPoleEnv
from pygame import gfxdraw


class CartPoleEnvCustom(CartPoleEnv):
    def __init__(
        self,
        sutton_barto_reward: bool = False,
        render_mode: str | None = None,
        threshold_degrees: int = 90,
    ):
        """
        Custom cart pole env that's more fun.

        Args:
        - threshold_degrees, when to fail the episode. they use 12, more fun with 90
          (falls all the way)
        """
        super().__init__(sutton_barto_reward, render_mode)

        self.theta_threshold_radians = threshold_degrees * 2 * math.pi / 360

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.screen_width = 800
        self.screen_height = 600

        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.step_number_font = pygame.font.Font(None, 48)

        self.step_number = 0

    def step(self, action):
        res = super().step(action)

        self.step_number += 1

        return res

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        res = super().reset(seed=seed, options=options)

        self.step_number = 0

        return res

    def _draw(self):
        """first half of parent's render() that actually draws"""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # OK to assume for now for us
        # try:
        #     import pygame
        #     from pygame import gfxdraw
        # except ImportError as e:
        #     raise DependencyNotInstalled(
        #         'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
        #     ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        self.scale = scale
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

    def _output(self):
        """second part of parent render() that actually outputs pixels"""
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def render(self):
        # draw base stuff
        self._draw()

        # draw our custom stuff
        center = self.screen_width / 2.0
        xlines = [-self.x_threshold, -2, -1, 0, 1, 2, self.x_threshold]

        cur_txt = self.step_number_font.render(str(self.step_number), True, (0, 0, 0))
        cur_txt = pygame.transform.flip(cur_txt, False, True)
        cur_text_rect = cur_txt.get_rect(center=(50, self.screen_height - 50))
        self.surf.blit(cur_txt, cur_text_rect)

        # Initialize font for text labels
        for i, xline in enumerate(xlines):
            x_pos = int(center + self.scale * xline)
            if i == 0:
                x_pos += 2
            elif i == len(xlines) - 1:
                x_pos -= 2

            gfxdraw.vline(self.surf, x_pos, 80, 120, (255, 0, 0))

            # Add centered text label
            text = self.font.render(str(xline), True, (255, 0, 0))
            text = pygame.transform.flip(text, False, True)

            if i == 0:
                x_pos += 8
            elif i == len(xlines) - 1:
                x_pos -= 8
            text_rect = text.get_rect(center=(x_pos, 130))
            self.surf.blit(text, text_rect)

        # output pixels
        return self._output()


gym.register(
    "CartPoleCustom-v0",
    "eval_env:CartPoleEnvCustom",
    max_episode_steps=5000,
    reward_threshold=5000.0,
)
