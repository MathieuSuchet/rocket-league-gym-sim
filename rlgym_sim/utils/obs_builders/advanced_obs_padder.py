import math
from typing import Any, List

import numpy as np

from rlgym_sim.utils import ObsBuilder, common_values
from rlgym_sim.utils.gamestates import PlayerData, GameState, PhysicsObject


class AdvancedObsPadder(ObsBuilder):
    POS_STD = 2300  # If you read this and wonder why, ping Rangler in the dead of night.
    ANG_STD = math.pi

    def __init__(self, max_team_size: int = 3):
        super().__init__()
        self.max_team_size = max_team_size

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        allies_padding_size = self.max_team_size - len([0 for p in state.players if p.team_num == player.team_num])
        enemies_padding_size = self.max_team_size - len([0 for p in state.players if p.team_num != player.team_num])

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        for _ in range(allies_padding_size):
            self._add_empty_player(allies)

        for _ in range(enemies_padding_size):
            self._add_empty_player(enemies)

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             player.passed_id,
             player.bumped_id,
             player.demoed_id,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed),
             int(player.has_flip_reset)]])

        return player_car

    def _add_empty_player(self, obs):
        obs.extend([[0] * 35])