import random
import torch
import numpy as np
from line_profiler_pycharm import profile
from pprint import pprint

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
                 reference_model_update_interval=5, max_replay_buffer_sample=200, device='gpu', silent=False):
        super(QLearningAI, self).__init__(game, player, silent)
        self.optimizer = None
        self.max_distance_from_enemy = max_distance_from_enemy

        self.device = device

        self.online_model = None
        self.reference_model = None

        self.lr = lr
        self.gamma = gamma
        self.reference_model_update_interval = reference_model_update_interval

        self.last_state = None
        self.last_action = None

        self.game_n = None

        self.model_lambda = lambda: nn.Sequential(
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

            self.online_model = self.model_lambda()
            self.reference_model = self.model_lambda()
            self.reference_model.load_state_dict(self.online_model.state_dict())

            self.replay_buffer = self.__replay_buffer_lambda()
        else:
            self.online_model, self.reference_model = models
            self.replay_buffer = replay_buffer

        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.online_model.to(self.device)
        self.reference_model.to(self.device)
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)

        self.game_n = game_n

        self.online_model.to(self.device)
        self.reference_model.to(self.device)

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

        state_tensor_batch = torch.stack(states).to(self.device).squeeze()
        action_tensor_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_tensor_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        next_state_values = torch.zeros(len(replay_buffer_sample), dtype=torch.float32).to(self.device)

        non_terminal_indices = [i for i, state in enumerate(next_states) if state is not None]

        non_terminal_next_states = [next_states[i] for i in non_terminal_indices]
        non_terminal_next_state_tensor_batch = torch.stack(non_terminal_next_states).to(self.device)

        non_terminal_next_state_values = self.reference_model(non_terminal_next_state_tensor_batch.squeeze())

        for idx, values in enumerate(non_terminal_next_state_values):
            legal_values = values[legal_actions_s_next[non_terminal_indices[idx]]]
            next_state_values[non_terminal_indices[idx]] = legal_values.max().item()

        reference_pred = reward_tensor_batch + self.gamma * next_state_values

        # ------------------------------------------------------------------------------------------
        # augment the input tensor with horizontal, vertical and horizontal-vertical flips
        state_tensor_batch_vflipped = torch.flip(state_tensor_batch, [2])
        state_tensor_batch_hflipped = torch.flip(state_tensor_batch, [3])
        state_tensor_batch_vhflipped = torch.flip(state_tensor_batch, [2, 3])
        state_tensor_batch = torch.cat((state_tensor_batch,
                                        state_tensor_batch_vflipped,
                                        state_tensor_batch_hflipped,
                                        state_tensor_batch_vhflipped), 0)

        # the reference_pred will not change under the flips:
        reference_pred = torch.cat((reference_pred,  reference_pred, reference_pred, reference_pred))

        # but actions will:
        vmapping = {0: 5, 5: 0, 2: 3, 3: 2, 1: 4, 4: 1, 6: 6}
        hmapping = {0: 2, 2: 0, 3: 5, 5: 3, 1: 1, 4: 4, 6: 6}

        action_tensor_batch_vflipped = torch.tensor([vmapping[val.item()] for val in action_tensor_batch])
        action_tensor_batch_hflipped = torch.tensor([hmapping[val.item()] for val in action_tensor_batch])
        action_tensor_batch_vhflipped = torch.tensor([hmapping[val.item()] for val in action_tensor_batch_vflipped])

        action_tensor_batch = torch.cat((action_tensor_batch,
                                         action_tensor_batch_vflipped,
                                         action_tensor_batch_hflipped,
                                         action_tensor_batch_vhflipped))
        # ------------------------------------------------------------------------------------------

        model_pred = self.online_model(state_tensor_batch).gather(1, action_tensor_batch.unsqueeze(-1)).squeeze(-1)

        loss = 1. / len(state_tensor_batch) * ((reference_pred - model_pred) ** 2).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

    def create_paths(self, queued_rewards):
        player = self.player

        log_prob_turn_total = torch.zeros(1)
        log_prob_turn_total.requires_grad = True

        if not self.silent:
            print(f'Units to create_paths for ({len(self.player.units)}):')
            pprint(self.player.units)
            print()

        for i, unit in reversed(list(enumerate(player.units))):
            if not self.silent:
                print(f'Creating paths for {unit} (turn {self.game.turn_number}). hp={unit.hp}, mp={unit.mp}')

            new_state = None
            new_state_legal_actions = None
            actions_taken_count = 1

            queued_rewards_dict = {}
            for entry in queued_rewards:
                unit_name = entry['to_unit']
                reward = {k: v for k, v in entry.items() if k != 'to_unit'}
                if unit_name not in queued_rewards_dict:
                    queued_rewards_dict[unit_name] = []
                queued_rewards_dict[unit_name].append(reward)

            destroyed_units = [unit for unit in queued_rewards_dict.keys() if unit not in self.player.units]
            if len(destroyed_units) and not self.silent:
                print('We have some units which were destroyed on the previous turn:')
                pprint(destroyed_units)
                print()
            
            for d_unit in destroyed_units:
                assert len(queued_rewards_dict[d_unit]) > 0
                
                if not self.silent:
                    print(f'Applying the queued rewards for (destoyed) unit {d_unit}:')
                    pprint(queued_rewards_dict[d_unit])

                self.replay_buffer.update_new_state_and_reward(
                    turn_number=self.game.turn_number - 1,
                    unit=unit,
                    new_state=None,
                    additional_reward=queued_rewards_dict[d_unit],
                    new_state_legal_action=None,)

            while True:
                if new_state is None:
                    assert new_state_legal_actions is None
                    state = self.create_input_tensor(unit).to(self.device)
                    legal_actions = self.get_legal_actions(unit)

                    if len(queued_rewards) > 0:
                        unit_queued_rewards = [r for r in queued_rewards if r['to_unit'] == unit]
                        
                        if not self.silent:
                            print(f'Applying the queued rewards for unit {unit}:')
                            pprint(unit_queued_rewards)
                            print()

                        self.replay_buffer.update_new_state_and_reward(
                            turn_number=self.game.turn_number - 1,
                            unit=unit,
                            new_state=state,
                            additional_reward=unit_queued_rewards,
                            new_state_legal_action=legal_actions,)

                else:
                    state = new_state

                    assert new_state_legal_actions is not None
                    legal_actions = new_state_legal_actions

                chosen_action = self.select_action(state, legal_actions)
                
                if chosen_action <= 5:
                    target_coords = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)[chosen_action]

                    if not self.silent:
                        print(f'{actions_taken_count}) {unit.name} {unit.coords} -> {target_coords}.'
                        f' (1/{len(legal_actions)} legal actions)')
                
                    reward = unit.move_one_cell(self.game, *target_coords, calc_rewards_for=[self.player])[self.player]
                else:
                    # just stay where we are x2
                    if not self.silent:
                        print(f'{actions_taken_count}) {unit.name} {unit.coords} -> {unit.coords} (stayed where he is).'
                          f' (1/{len(legal_actions)} legal actions)')
                    unit.path = []

                    target_coords = unit.coords

                    reward = [Rewards.get_named_reward(Rewards.STAYING_WHERE_HE_IS, to_unit='self')]

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
