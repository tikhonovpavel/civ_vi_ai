import random

import torch
import numpy as np
from line_profiler_pycharm import profile

import torch.optim as optim
import torch.nn as nn

from replay_buffer import ReplayBuffer, Transition
from rewards_values import Rewards
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


class QLearningAI(TrainableAI):
    def __init__(self, game, player, max_distance_from_enemy=3, lr=0.01, gamma=0.98,
                 reference_model_update_interval=5, max_replay_buffer_sample=200, device='cpu', silent=False):
        super(QLearningAI, self).__init__(game, player, silent)
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
        self.__replay_buffer_lambda = lambda: ReplayBuffer(capacity=500)

        self.replay_buffer = None
        self.max_replay_buffer_sample = max_replay_buffer_sample

    def init(self, game_n, *init_objects):
        models, replay_buffer = init_objects

        if models is None:
            assert replay_buffer is None

            self.online_model = self.__model_lambda()
            self.reference_model = self.__model_lambda()
            self.reference_model.load_state_dict(self.online_model.state_dict())

            self.replay_buffer = self.__replay_buffer_lambda()
        else:
            self.online_model, self.reference_model = models
            self.replay_buffer = replay_buffer
            # self.reference_model = self.__model_lambda()
            #
            # self.reference_model.load_state_dict(self.online_model.state_dict())

        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.online_model.to(self.device)
        self.reference_model.to(self.device)
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)

        self.game_n = game_n

        return (self.online_model, self.reference_model), self.replay_buffer

    def update_models(self):
        sample_size = min(max(len(self.replay_buffer.buffer) // 10, self.max_replay_buffer_sample), len(self.replay_buffer.buffer))
        replay_buffer_sample = self.replay_buffer.sample(sample_size)

        if not self.silent:
            print(f'Replay Buffer sample size: {len(replay_buffer_sample)}')

        states = [transition.s for transition in replay_buffer_sample]
        actions = [transition.a for transition in replay_buffer_sample]
        rewards = [transition.total_reward for transition in replay_buffer_sample]
        next_states = [transition.s_next for transition in replay_buffer_sample]
        legal_actions_s_next = [transition.legal_actions_s_next for transition in replay_buffer_sample]

        state_tensor_batch = torch.stack(states).to(self.device)
        action_tensor_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_tensor_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Изначально все next_state_values установим в 0
        next_state_values = torch.zeros(len(replay_buffer_sample), dtype=torch.float32).to(self.device)

        # Найдем индексы не-None состояний и сохраним их
        non_terminal_indices = [i for i, state in enumerate(next_states) if state is not None]

        # Теперь создадим тензор только из не-None состояний
        non_terminal_next_states = [next_states[i] for i in non_terminal_indices]
        non_terminal_next_state_tensor_batch = torch.stack(non_terminal_next_states).to(self.device)

        # Вычисляем Q-значения для всех не-None следующих состояний
        non_terminal_next_state_values = self.reference_model(non_terminal_next_state_tensor_batch.squeeze())

        for idx, values in enumerate(non_terminal_next_state_values):
            legal_values = values[legal_actions_s_next[non_terminal_indices[idx]]]
            next_state_values[non_terminal_indices[idx]] = legal_values.max().item()

        reference_pred = reward_tensor_batch + self.gamma * next_state_values

        # Для Q-сети нам нужно предсказать Q-значение для совершенного действия
        model_pred = self.online_model(state_tensor_batch.squeeze()).gather(1, action_tensor_batch.unsqueeze(-1)).squeeze(-1)

        loss = 1. / len(replay_buffer_sample) * ((reference_pred - model_pred) ** 2).sum()

        # Обратное распространение ошибки
        self.optimizer.zero_grad()  # Обнуляем градиенты
        loss.backward()  # Вычисляем градиенты
        self.optimizer.step()  # Обновляем веса

        if self.game_n % self.reference_model_update_interval == 0:
            self.reference_model.load_state_dict(self.online_model.state_dict())

    def get_legal_actions(self, unit):
        """
        Legal actions are: 
            * 1st-6th - are nearest neighbours
            * 7th - a cell, where unit already are
        """
        nearest_neighbours = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)

        return [i for i, cell in enumerate(nearest_neighbours)
                if self.check_if_legal(unit, cell, nearest_neighbours)]  + [6]  # staying where you are is always legal

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

        if not self.silent:
            print(f'Units to create_paths for ({len(self.player.units)}): {self.player.units}')
        for i, unit in reversed(list(enumerate(player.units))):

            if not self.silent:
                print(f'Creating paths for <{unit}> (turn {self.game.turn_number}). hp={unit.hp}, mp={unit.mp}')

            new_state = None
            new_state_legal_actions = None
            actions_taken_count = 1

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
                            additional_reward=Rewards.get_named_reward(Rewards.SURVIVED_THE_TURN),
                            new_state_legal_action=legal_actions,)

                else:
                    state = new_state

                    assert new_state_legal_actions is not None
                    legal_actions = new_state_legal_actions

                # if len(legal_actions) == 0:
                #     # Если так получилось, то это самое первое действие юнита на этом ходу, а ему уже некуда идти
                #     # значит е

                chosen_action = self.select_action(state, legal_actions)
                # chosen_action_old = chosen_action
                # chosen_action = legal_cells[chosen_action]

                if chosen_action <= 5:
                    target_coords = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)[chosen_action]

                    if not self.silent:
                        print(f'{actions_taken_count}) {unit.name} {unit.coords} -> {target_coords}.'
                        f' (1/{len(legal_actions)} legal actions)')
                
                    reward = unit.move_one_cell(self.game, *target_coords, calc_rewards_for=[self.player])
                else:
                    # just stay where we are x2
                    if not self.silent:
                        print(f'{actions_taken_count}) {unit.name} {unit.coords} -> {unit.coords} (stayed where he is).'
                          f' (1/{len(legal_actions)} legal actions)')
                    unit.path = []

                    target_coords = unit.coords

                    reward = [Rewards.get_named_reward(Rewards.STAYING_WHERE_HE_IS)]

                    # ну что ж, выбор оставаться на месте - окончательный. 
                    # Даже если у юнита остались ОП, мы на это забиваем, и переходим к следующему юниту
                    player.ai.replay_buffer.add(Transition(game_number=self.game_n,
                                                           turn_number=self.game.turn_number,
                                                           movement_number=actions_taken_count,
                                                           unit=unit,
                                                           s=state,
                                                           a=chosen_action,
                                                           r=reward,
                                                           s_next=None,
                                                           legal_actions_s_next=None))
                    break

                self.game.update()

                if unit.mp != 0 and unit.hp:  # then just continue to move this unit
                    new_state = self.create_input_tensor(unit).to(self.device)
                    new_state_legal_actions = self.get_legal_actions(unit)

                    if len(new_state_legal_actions) == 0:
                        # юнит дурак и забрёл не пойми куда. Больше он не может ходить на этом тёрне, хотя и остались очки перемещения
                        # поэтому нам надо оставлять его с None стейтом
                        player.ai.replay_buffer.add(Transition(game_number=self.game_n,
                                                               turn_number=self.game.turn_number,
                                                               movement_number=actions_taken_count,
                                                               unit=unit,
                                                               s=state,
                                                               a=chosen_action,
                                                               r=reward,
                                                               s_next=None,
                                                               legal_actions_s_next=None))
                        break

                    player.ai.replay_buffer.add(Transition(game_number=self.game_n,
                                                           turn_number=self.game.turn_number,
                                                           movement_number=actions_taken_count,
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
                                                           movement_number=actions_taken_count,
                                                           unit=unit,
                                                           s=state,
                                                           a=chosen_action,
                                                           r=reward,
                                                           s_next=None,
                                                           legal_actions_s_next=None))

                    break

                actions_taken_count += 1


            if not self.silent:
                print(f'{i + 1}/{len(player.units)} Unit {unit.category} {unit.name} done. Took {actions_taken_count} steps')
            # print()
