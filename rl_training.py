import math
import random
from dataclasses import dataclass
from typing import List

import torch
import numpy as np
from line_profiler_pycharm import profile

import torch.optim as optim

import torch.nn as nn

import rewards_values
from ai import TrainableAI

# playing the following game:

# AI algorithm plays vs 2nd AI driven by the SimpleAI witch is just attacks the nearest enemy unit or city
# There is also a 3rd player in the middle of the map, controlled by the SimpleAIHikikomori.
# It's only attacks the nearest unit if it is within 3 cells of SimpleAIHikikomori's homeland;
# otherwise, it returns to its homeland.
#
# The winning condition is to capture the only city of the 3rd player
from game_object import MilitaryObject
from unit import Unit

random.seed(42)
torch.manual_seed(42)



class ResidualLayer(torch.nn.Module):
    def __init__(self, n_channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class PolarCoordinatesLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        rho, phi = torch.split(x, 1, dim=1)
        phi = torch.sigmoid(phi) * 2 * np.pi  # to range [0, 2*pi]
        rho = torch.exp(rho)
        return torch.cat((rho, phi), dim=1)


@dataclass
class Transition:
    game_number: int
    turn_number: int
    unit: Unit
    s: torch.Tensor
    a: int
    r: int
    s_next: torch.Tensor | None
    legal_actions_s_next: List[int] | None


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []

    def add(self, transition: Transition):
        if len(self.buffer) > 30:
            print()
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def update_new_state_and_reward(self, turn_number, unit, new_state, new_state_legal_action, additional_reward):
        """
        Updates the state and reward for the specified unit and turn.

        Parameters:
        - turn_number (int): The turn number for which the information needs to be updated.
        - unit (Unit): The unit for which the information needs to be updated.
        - new_state (Tensor): The new state to set.
        - additional_reward (float): Additional reward to be added to the existing one.

        Raises an exception if:
        - No corresponding transition for update is found.
        - More than one candidate for update is found.
        """

        candidates_count = 0
        updated = False
        for transition in reversed(self.buffer):
            if transition.turn_number == turn_number and transition.unit == unit and transition.s_next is None:
                candidates_count += 1
                if candidates_count > 1:
                    raise ValueError(f"Multiple candidates found for unit {unit} "
                                     f"on turn {turn_number} with s_next as None.")

                assert transition.s_next is None and transition.legal_actions_s_next is None

                transition.s_next = new_state
                transition.legal_actions_s_next = new_state_legal_action
                transition.r += additional_reward
                updated = True
                break

        if not updated:
            raise ValueError(f"No transition found for unit {unit} on turn {turn_number} with s_next as None.")


class QLearningAI(TrainableAI):
    def __init__(self, game, player, max_distance_from_enemy=3, lr=0.01, gamma=0.98,
                 reference_model_update_interval=5, device='cpu'):
        super(QLearningAI, self).__init__(game, player)
        self.optimizer = None
        self.max_distance_from_enemy = max_distance_from_enemy

        self.device = device

        self.std_rho = 1.0  # Tune this value based on your game
        self.std_phi = 0.1  # Tune this value based on your game

        self.online_model = None
        self.reference_model = None

        self.lr = lr
        self.gamma = gamma
        self.reference_model_update_interval = reference_model_update_interval

        self.last_state = None
        self.last_action = None

        self.game_n = None

        self.__model_lambda = lambda: nn.Sequential(
            nn.Conv2d(17, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

        self.replay_buffer = ReplayBuffer(capacity=500)

    def init_models(self, models, game_n):
        if models is None:
            self.online_model = self.__model_lambda()
            self.reference_model = self.__model_lambda()

            self.reference_model.load_state_dict(self.online_model.state_dict())
        else:
            self.online_model, self.reference_model = models
            # self.reference_model = self.__model_lambda()
            #
            # self.reference_model.load_state_dict(self.online_model.state_dict())

        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.online_model.to(self.device)
        self.reference_model.to(self.device)
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)

        self.game_n = game_n

        return self.online_model, self.reference_model

    def update_models(self):
        replay_buffer_sample = self.replay_buffer.sample(len(self.replay_buffer.buffer) // 10)
        print(f'Replay Buffer sample size: {len(replay_buffer_sample)}')

        states = [transition.s for transition in replay_buffer_sample]
        actions = [transition.a for transition in replay_buffer_sample]
        rewards = [transition.r for transition in replay_buffer_sample]
        next_states = [transition.s_next for transition in replay_buffer_sample]

        # Извлекаем legal_actions для следующего состояния из replay_buffer_sample
        legal_actions_s_next = [transition.legal_actions_s_next for transition in replay_buffer_sample]

        state_tensor_batch = torch.stack(states).to(self.device)
        action_tensor_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_tensor_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_tensor_batch = torch.stack([s for s in next_states if s is not None]).to(self.device)

        # Для Q-сети нам нужно предсказать Q-значение для совершенного действия
        model_pred = self.online_model(state_tensor_batch.squeeze()).gather(1, action_tensor_batch.unsqueeze(-1)).squeeze(-1)

        # Для следующего состояния мы выбираем максимальное Q-значение только среди допустимых действий
        all_next_state_values = self.reference_model(next_state_tensor_batch.squeeze())
        next_state_values = []
        for idx, values in enumerate(all_next_state_values):
            legal_values = values[legal_actions_s_next[idx]]
            next_state_values.append(legal_values.max().item())
        next_state_values = torch.tensor(next_state_values, dtype=torch.float32).detach().to(self.device)
        reference_pred = reward_tensor_batch + self.gamma * next_state_values

        loss = 1. / len(replay_buffer_sample) * ((reference_pred - model_pred) ** 2).sum()

        # Обратное распространение ошибки
        self.optimizer.zero_grad()  # Обнуляем градиенты
        loss.backward()  # Вычисляем градиенты
        self.optimizer.step()  # Обновляем веса

        if self.game_n % self.reference_model_update_interval == 0:
            self.reference_model.load_state_dict(self.online_model.state_dict())

    def get_legal_actions(self, unit):
        nearest_neighbours = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)

        return [i for i, cell in enumerate(nearest_neighbours)
                if self.check_if_legal(unit, cell, nearest_neighbours)]

    def select_action(self, state, legal_actions, epsilon=0.1):
        # Прогоняем состояние через Q-сеть
        q_values = self.online_model(state)

        # Оставляем только Q-значения для легальных действий
        legal_q_values = q_values[0, legal_actions]

        # Если случайное число меньше epsilon, выбираем случайное действие из легальных
        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            # Иначе выбираем легальное действие с максимальным Q-значением
            action_idx = legal_q_values.argmax().item()  # индекс в массиве legal_q_values
            action = legal_actions[action_idx]  # соответствующее действие из списка легальных действий

        return action

    @profile
    def check_if_legal(self, unit, cell, nearest_neighbours):
        player = self.player

        if cell is None:
            return False

        is_enemy_on_hex = len(self.game.get_game_objects_on_hex(*cell, only_enemies=True)) > 0
        cell_obj = self.game.map.get(*cell)
        cost = cell_obj.geometry.terrain.cost

        if cell not in nearest_neighbours:
            if unit.role == MilitaryObject.COMBAT:
                return False
            elif unit.role == MilitaryObject.RANGED:
                return is_enemy_on_hex and cost <= unit.mp_base and unit.can_attack
            else:
                raise NotImplementedError()

        if cost > unit.mp:
            if unit.role == MilitaryObject.COMBAT:
                return False

            # ranged can go, only if there is an enemy on the cell and we can attack
            return is_enemy_on_hex and unit.can_attack

        # if we are here, then cell is in nearest neighbours and cost < mp

        if unit.role == MilitaryObject.COMBAT and is_enemy_on_hex:
            return unit.can_attack

        # false only if unit on the cell is of the same nation as the player
        return len(cell_obj.game_objects) == 0 or cell_obj.game_objects[0].player.nation != player.nation

    def create_paths(self):
        player = self.player

        log_prob_turn_total = torch.zeros(1)
        log_prob_turn_total.requires_grad = True

        for i, unit in enumerate(player.units):

            new_state = None
            new_state_legal_actions = None

            actions_taken_count = 0
            while True:
                if new_state is None:
                    assert new_state_legal_actions is None
                    state = self.create_input_tensor(unit).to(self.device)
                    legal_actions = self.get_legal_actions(unit)

                    # если это первый ход, то значит этого товарища ещё нет в реплей баффере, поэтому ничего не делаем
                    #
                    # но если это не первый ход внутри одной игры, то тогда последнее состояние юнита в баффере должно быть None
                    # и мы должны заменить его на текущее
                    #
                    # Note: если этого товарища убил противник своей атакой, то это был бы особый случай. Но поскольку тогда
                    # юнит удаляется из списка player.units, то этот случай получается здесь не надо это учитывать,
                    # и обработка уже произошла в методе destroy в момент убийства

                    if self.game.turn_number > 1:
                        self.replay_buffer.update_new_state_and_reward(
                            turn_number=self.game.turn_number - 1,
                            unit=unit,
                            new_state=state,
                            additional_reward=0,
                            new_state_legal_action=legal_actions,)

                else:
                    state = new_state

                    assert new_state_legal_actions is not None
                    legal_actions = new_state_legal_actions

                if len(legal_actions) == 0:
                    # just stay where we are
                    unit.path = []
                    break

                chosen_action = self.select_action(state, legal_actions)
                # chosen_action_old = chosen_action
                # chosen_action = legal_cells[chosen_action]

                if chosen_action < 6:
                    target_coords = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)[chosen_action]

                    # if target_coords == unit_init_coords:
                    #     player.add_reward(rewards_values.RETURNED_TO_THE_INIT_POSITION)
                else:
                    # just stay where we are x2
                    print(f'{actions_taken_count + 1}) {unit.name} {unit.coords} -> {unit.coords} (stayed where he is).'
                          # f' prob was {legal_action_probabilities[chosen_action_old].item()}'
                          f' (1/{len(legal_actions)} actions)')
                    unit.path = []
                    break

                print(f'{actions_taken_count + 1}) {unit.name} {unit.coords} -> {target_coords}.'
                      # f' prob was {legal_action_probabilities[chosen_action_old].item()}'
                      f' (1/{len(legal_actions)} actions)')
                reward = unit.move_one_cell(self.game, *target_coords)
                self.game.update()
                actions_taken_count += 1

                if unit.mp != 0 and unit.hp:  # then just continue to move this unit
                    new_state = self.create_input_tensor(unit).to(self.device)
                    new_state_legal_actions = self.get_legal_actions(unit)

                    if len(new_state_legal_actions) == 0:
                        print()

                    player.ai.replay_buffer.add(Transition(game_number=self.game_n,
                                                           turn_number=self.game.turn_number,
                                                           unit=unit,
                                                           s=state,
                                                           a=chosen_action,
                                                           r=reward,
                                                           s_next=new_state,
                                                           legal_actions_s_next=new_state_legal_actions))
                else:
                    # no more moves left for the unit. We don't yet know the next state
                    # (and therefore the legal_actions at this state), so leave them as None
                    # unit.mp = unit.mp_base

                    player.ai.replay_buffer.add(Transition(game_number=self.game_n,
                                                           turn_number=self.game.turn_number,
                                                           unit=unit,
                                                           s=state,
                                                           a=chosen_action,
                                                           r=reward,
                                                           s_next=None,
                                                           legal_actions_s_next=None))

                    break

            print(f'{i + 1}/{len(player.units)} Unit {unit.category} {unit.name} done. Took {actions_taken_count} steps')
            # print()


# class PolicyGradientAI(TrainableAI):
#     def __init__(self, game, player, max_distance_from_enemy=3, lr=0.01, device='cpu', policy_network=None):
#         super(PolicyGradientAI, self).__init__(game, player)
#         self.max_distance_from_enemy = max_distance_from_enemy
#
#         self.device = device
#
#         self.std_rho = 1.0  # Tune this value based on your game
#         self.std_phi = 0.1  # Tune this value based on your game
#
#         self.policy_network = None
#         self.lr = lr
#
#         # self.mu_rho, self.log_sigma_rho, self.mu_phi, self.log_sigma_phi = \
#         #     torch.chunk(self.policy_network[-1].output, 4, dim=1)
#
#         # self.policy_network[-1].register_forward_hook(self.polar_coordinate_activations)
#
#         # self.policy_network = torch.nn.Sequential(
#         #     torch.nn.Conv2d(14, 24, kernel_size=3, stride=1, padding=1),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Flatten(),
#         #     torch.nn.Linear(24 * game.map.n_rows * game.map.n_columns, game.map.n_rows * game.map.n_columns),
#         #     torch.nn.Softmax(dim=-1)
#         # )
#         # self.states = []
#         # self.actions = []
#         self.rewards = []
#         self._rewards_history = []
#         self.saved_log_probs = []
#
#     def init_policy_network(self, policy_network=None):
#         if policy_network is None:
#             self.policy_network = torch.nn.Sequential(
#                 torch.nn.Conv2d(17, 24, kernel_size=3, stride=1, padding=1),
#                 torch.nn.ReLU(),
#                 ResidualLayer(24),
#                 ResidualLayer(24),
#                 torch.nn.Flatten(),
#                 torch.nn.Linear(24 * self.game.map.n_rows * self.game.map.n_columns, 19),
#                 # torch.nn.Softmax(dim=-1),
#                 # PolarCoordinatesLayer()
#             )
#         else:
#             self.policy_network = policy_network
#
#         self.policy_network.to(self.device)
#         self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
#
#         return policy_network
#
#
#     @staticmethod
#     def polar_coordinate_activations(_, __, output):
#         rho, phi = output.chunk(2, dim=-1)
#         rho = torch.nn.functional.relu(rho)
#         phi = (torch.tanh(phi) + 1) * np.pi
#         return torch.cat([rho, phi], dim=-1)
#
#     def get_target_rc_from_action(self, action, unit):
#         r = action // self.game.map.n_rows
#         c = action % self.game.map.n_columns
#
#         return r, c
#
#     def polar_to_cartesian(self, rho, phi):
#         x = rho * torch.cos(phi).item()
#         y = rho * torch.sin(phi).item()
#
#         return x, y
#
#     @profile
#     def check_if_legal(self, unit, cell, nearest_neighbours):
#         player = self.player
#
#         if cell is None:
#             return False
#
#         is_enemy_on_hex = len(self.game.get_game_objects_on_hex(*cell, only_enemies=True)) > 0
#         cell_obj = self.game.map.get(*cell)
#         cost = cell_obj.geometry.terrain.cost
#
#         if cell not in nearest_neighbours:
#             if unit.role == MilitaryObject.COMBAT:
#                 return False
#             elif unit.role == MilitaryObject.RANGED:
#                 return is_enemy_on_hex and cost <= unit.mp_base and unit.can_attack
#             else:
#                 raise NotImplementedError()
#
#         if cost > unit.mp:
#             if unit.role == MilitaryObject.COMBAT:
#                 return False
#
#             # ranged can go, only if there is an enemy on the cell and we can attack
#             return is_enemy_on_hex and unit.can_attack
#
#         # if we are here, then cell is in nearest neighbours and cost < mp
#
#         if unit.role == MilitaryObject.COMBAT and is_enemy_on_hex:
#             return unit.can_attack
#
#         # false only if unit on the cell is of the same nation as the player
#         return len(cell_obj.game_objects) == 0 or cell_obj.game_objects[0].player.nation != player.nation
#
#
#         # if is_enemy_on_hex:
#         #     if unit.category == MilitaryObject.COMBAT and not unit.can_attack:
#         #         return False
#         #     elif unit.category == MilitaryObject.RANGED:
#         #         return True
#         # else:
#         #     if cell.geometry.terrain.cost > unit.mp:
#         #         return False
#
#
#
#     # def get_actions_list(self, r, c):
#     #     nearest_neighbours = self.game.map.get_neighbours_grid_coords(r, c)
#     #
#     #     result = nearest_neighbours.copy()
#     #
#     #     for n_neighbour in nearest_neighbours:
#     #         far_neighbours = self.game.map.get_neighbours_grid_coords(*n_neighbour)
#     #
#     #         for f_neighbour in far_neighbours:
#     #             if f_neighbour not in result:
#     #                 result.append(f_neighbour)
#     #
#     #     return result
#
#     @profile
#     def create_paths(self):
#         player = self.player
#
#         if len(self.rewards) != len(self.saved_log_probs):
#             raise Exception('You should call receive_reward after create_paths at each turn')
#
#         log_prob_turn_total = torch.zeros(1)
#         log_prob_turn_total.requires_grad = True
#
#         try:
#             input_tensor = self.create_input_tensor().to(self.device)
#         except RuntimeError as err:
#             if 'stack expects a non-empty TensorList' not in err.args:
#                 raise err
#
#             self.saved_log_probs.append(log_prob_turn_total)
#             return
#
#         for i, unit in enumerate(player.units):
#             unit_init_coords = unit.r, unit.c
#
#             count = 0
#             # if unit.name == 'Shell Shock':
#             #     return
#                 # print(unit.name)
#
#             while unit.mp != 0 and unit.hp > 0:
#                 action_logits = self.policy_network(input_tensor[i].unsqueeze(0))
#                 # try:
#                 #     action_logits = self.policy_network(input_tensor[i].unsqueeze(0))
#                 # except Exception as err:
#                 #     self.create_input_tensor().to(self.device)
#                 #     raise err
#
#                 all_neighbour_cells = self.game.map.get_neighbours_grid_coords(unit.r, unit.c, radius=2)
#                 nearest_neighbours = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)
#
#                 legal_cells = [i for i, cell in enumerate(all_neighbour_cells)
#                                if self.check_if_legal(unit, cell, nearest_neighbours)]
#
#                 if len(legal_cells) == 0:
#                     # just stay where we are
#                     unit.path = []
#                     break
#
#                 # Get the logits corresponding to legal cells
#                 legal_action_logits = action_logits[0, legal_cells]
#
#                 # Calculate the softmax values of these logits
#                 legal_action_probabilities = torch.nn.functional.softmax(legal_action_logits, dim=0)
#
#                 # Now, we can sample directly from the legal_action_probabilities
#                 action_distribution = torch.distributions.Categorical(legal_action_probabilities)
#                 chosen_action = action_distribution.sample().item()
#
#                 log_prob_turn_total = log_prob_turn_total + torch.log(legal_action_probabilities[chosen_action])
#
#                 # Since chosen_action is an index of legal_action_probabilities,
#                 # we need to map it back to the corresponding action
#                 chosen_action_old = chosen_action
#                 chosen_action = legal_cells[chosen_action]
#
#                 if chosen_action < 6:
#                     target_coords = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)[chosen_action]
#
#                     if target_coords == unit_init_coords:
#                         player.add_reward(rewards_values.RETURNED_TO_THE_INIT_POSITION)
#                 else:
#                     # just stay where we are x2
#                     print(f'{count + 1}) {unit.name} {unit.coords} -> {unit.coords} (stayed where he is).'
#                           f' prob was {legal_action_probabilities[chosen_action_old].item()}'
#                           f' (1/{len(legal_cells)} actions)')
#                     unit.path = []
#                     break
#
#                 print(f'{count + 1}) {unit.name} {unit.coords} -> {target_coords}.'
#                       f' prob was {legal_action_probabilities[chosen_action_old].item()}'
#                       f' (1/{len(legal_cells)} actions)')
#                 unit.move_one_cell(self.game, *target_coords)
#                 self.game.update()
#                 count += 1
#
#             print(f'{i+1}/{len(player.units)} Unit {unit.category} {unit.name} done. Took {count} steps')
#
#
#         self.saved_log_probs.append(log_prob_turn_total)
