import math

import torch
import numpy as np
from line_profiler_pycharm import profile

import torch.optim as optim

import rewards_values
from ai import TrainableAI

# playing the following game:

# AI algorithm plays vs 2nd AI driven by the SimpleAI witch is just attacks the nearest enemy unit or city
# There is also a 3rd player in the middle of the map, controlled by the SimpleAIHikikomori.
# It's only attacks the nearest unit if it is within 3 cells of SimpleAIHikikomori's homeland;
# otherwise, it returns to its homeland.
#
# The winning condition is to capture the only city of the 3rd player

torch.manual_seed(1337)


terrain_classes = ['plains', 'hills', 'forest', 'water']
units_classes = [None, 'artillery', 'tank']
cities_classes = [None, 'city']

terrains_label_encoder = {cat: i for i, cat in enumerate(terrain_classes)}


@profile
def get_attribute_map(collection, attr_name, n_rows, n_cols, classes):
    label_encoder = {cat: i for i, cat in enumerate(classes)}

    attribute_map = torch.zeros((n_rows, n_cols), dtype=torch.int64)
    for obj in collection:
        attribute_map[obj.r][obj.c] = label_encoder[getattr(obj, attr_name)]

    return attribute_map


@profile
def to_onehot_tensor(attribute_map, num_classes):
    # label_encoder = {cat: i for i, cat in enumerate(classes)}
    # labels = np.vectorize(label_encoder.get)(attribute_map)
    # labels = torch.LongTensor(labels)

    attribute_map = torch.LongTensor(attribute_map)

    onehot = torch.nn.functional.one_hot(attribute_map, num_classes=num_classes)
    onehot = onehot.permute(2, 0, 1)

    return onehot

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


class PolicyGradientAI(TrainableAI):
    def __init__(self, game, player, max_distance_from_enemy=3, lr=0.01, device='cpu'):
        super(PolicyGradientAI, self).__init__(game, player)
        self.max_distance_from_enemy = max_distance_from_enemy

        self.device = device

        self.std_rho = 1.0  # Tune this value based on your game
        self.std_phi = 0.1  # Tune this value based on your game

        self.policy_network = torch.nn.Sequential(
            torch.nn.Conv2d(17, 24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            ResidualLayer(24),
            ResidualLayer(24),
            torch.nn.Flatten(),
            torch.nn.Linear(24 * game.map.n_rows * game.map.n_columns, 7),
            # torch.nn.Softmax(dim=-1),
            # PolarCoordinatesLayer()
        )
        # self.mu_rho, self.log_sigma_rho, self.mu_phi, self.log_sigma_phi = \
        #     torch.chunk(self.policy_network[-1].output, 4, dim=1)

        # self.policy_network[-1].register_forward_hook(self.polar_coordinate_activations)

        # self.policy_network = torch.nn.Sequential(
        #     torch.nn.Conv2d(14, 24, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(24 * game.map.n_rows * game.map.n_columns, game.map.n_rows * game.map.n_columns),
        #     torch.nn.Softmax(dim=-1)
        # )
        self.policy_network.to(device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        # self.states = []
        # self.actions = []
        self.rewards = []
        self.saved_log_probs = []

    @staticmethod
    def polar_coordinate_activations(_, __, output):
        rho, phi = output.chunk(2, dim=-1)
        rho = torch.nn.functional.relu(rho)
        phi = (torch.tanh(phi) + 1) * np.pi
        return torch.cat([rho, phi], dim=-1)

    # def select_action(self, state):
    #     probabilities = self.policy_network(state)
    #     action = torch.distributions.Categorical(probabilities).sample()
    #
    #     return action

    @profile
    def update_policy(self, gamma=0.99):
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        self.optimizer.zero_grad()
        for log_prob, r in zip(self.saved_log_probs, returns):
            loss = -log_prob * r
            loss.backward()
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards = []

    @profile
    def convert_map_to_tensor(self):
        player = self.player
        game = self.game

        all_units = player.units + sum([p.units for p in game.diplomacy.get_enemies(player)], [])
        all_cities = player.cities + sum([p.cities for p in game.diplomacy.get_enemies(player)], [])

        terrains = game.map.get_terrains_map()
        terrains = [[terrains_label_encoder[name] for name in row] for row in terrains]
        terrains = torch.LongTensor(terrains)
        terrains_onehot = to_onehot_tensor(terrains, num_classes=len(terrain_classes))
        # print(f"Shape of terrains_onehot: {terrains_onehot.shape}")

        my_units = get_attribute_map(player.units, 'category', game.map.n_rows, game.map.n_columns, units_classes)
        my_units_onehot = to_onehot_tensor(my_units, num_classes=len(units_classes))
        # print(f"Shape of my_units_onehot: {my_units_onehot.shape}")

        my_cities = get_attribute_map(player.cities, 'category', game.map.n_rows, game.map.n_columns, cities_classes)
        my_cities_onehot = to_onehot_tensor(my_cities, num_classes=len(cities_classes))
        my_cities_onehot = my_cities_onehot[1:]
        # print(f"Shape of my_cities_onehot: {my_cities_onehot.shape}")

        enemy_units = get_attribute_map(sum((p.units for p in game.diplomacy.get_enemies(player)), []), 'category',
                                        game.map.n_rows, game.map.n_columns, units_classes)
        enemy_units_onehot = to_onehot_tensor(enemy_units, num_classes=len(units_classes))
        # print(f"Shape of enemy_units_onehot: {enemy_units_onehot.shape}")

        enemy_cities = get_attribute_map(sum((p.cities for p in game.diplomacy.get_enemies(player)), []), 'category',
                                         game.map.n_rows, game.map.n_columns, cities_classes)
        enemy_cities_onehot = to_onehot_tensor(enemy_cities, num_classes=len(cities_classes))
        enemy_cities_onehot = enemy_cities_onehot[1:]
        # print(f"Shape of enemy_cities_onehot: {enemy_cities_onehot.shape}")

        units_hp = np.zeros((game.map.n_rows, game.map.n_columns))
        for u in all_units:
            units_hp[u.r][u.c] = u.hp
        units_hp = torch.tensor(units_hp).unsqueeze(0)
        # print(f"Shape of units_hp: {units_hp.shape}")

        cities_hp = np.zeros((game.map.n_rows, game.map.n_columns))
        for city in all_cities:
            cities_hp[city.r][city.c] = city.hp
        cities_hp = torch.tensor(cities_hp).unsqueeze(0)
        # print(f"Shape of cities_hp: {cities_hp.shape}")

        units_mp = np.zeros((game.map.n_rows, game.map.n_columns))
        for u in all_units:
            units_mp[u.r][u.c] = u.mp
        units_mp = torch.tensor(units_mp).unsqueeze(0)
        # print(f"Shape of units_mp: {units_mp.shape}")

        result = torch.vstack([terrains_onehot,
                               my_units_onehot, my_cities_onehot, enemy_units_onehot, enemy_cities_onehot,
                               units_hp, cities_hp, units_mp])
        # print(f"Shape of result: {result.shape}")

        return result

    @profile
    def create_input_tensor(self):
        """
        :return: (n_units, n_layers, map_n_rows, map_n_columns)
        """
        player = self.player

        map_tensor = self.convert_map_to_tensor()

        result = []

        for unit in player.units:
            current_unit_layer = torch.zeros((self.game.map.n_rows, self.game.map.n_columns))
            current_unit_layer[unit.r][unit.c] = 1
            current_unit_layer = current_unit_layer.unsqueeze(0)

            reachable_cells_layer = torch.zeros((self.game.map.n_rows, self.game.map.n_columns))
            # reachable_cells = unit.get_reachable_cells(self.game)
            # indices = torch.tensor(reachable_cells)
            # reachable_cells_layer[indices[:, 0], indices[:, 1]] = 1
            reachable_cells_layer = reachable_cells_layer.unsqueeze(0)

            result.append(torch.vstack([map_tensor, current_unit_layer, reachable_cells_layer]))

        result = torch.stack(result)

        return result.float()

    def receive_reward(self, reward):
        self.rewards.append(reward)

        # Update the policy if there's enough data
        if len(self.rewards) > 5:
            self.update_policy()

    def get_target_rc_from_action(self, action, unit):
        r = action // self.game.map.n_rows
        c = action % self.game.map.n_columns

        return r, c

    def polar_to_cartesian(self, rho, phi):
        x = rho * torch.cos(phi).item()
        y = rho * torch.sin(phi).item()

        return x, y

    def check_if_legal(self, cell):
        player = self.player

        if cell is None:
            return False

        objs = self.game.map.get(*cell).game_objects

        # false only if unit on the cell is of the same nation as the player
        return len(objs) == 0 or self.game.map.get(*cell).game_objects[0].player.nation != player.nation


    @profile
    def create_paths(self):
        player = self.player

        if len(self.rewards) != len(self.saved_log_probs):
            raise Exception('You should call receive_reward after create_paths at each turn')

        input_tensor = self.create_input_tensor().to(self.device)

        log_prob_turn_total = 0
        for i, unit in enumerate(player.units):

            unit_init_coords = unit.r, unit.c

            # if unit.name == 'Siege Sentinel':
            #     print(unit.name)

            while unit.mp != 0 and unit.hp > 0:
                try:
                    action_logits = self.policy_network(input_tensor[i].unsqueeze(0))
                except Exception as err:
                    self.create_input_tensor().to(self.device)
                    raise err

                neighbour_cells = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)

                legal_cells = [i for i, cell in enumerate(neighbour_cells) if self.check_if_legal(cell)]

                # Get the logits corresponding to legal cells
                legal_action_logits = action_logits[0, legal_cells]

                # Calculate the softmax values of these logits
                legal_action_probabilities = torch.nn.functional.softmax(legal_action_logits, dim=0)

                # Now, we can sample directly from the legal_action_probabilities
                action_distribution = torch.distributions.Categorical(legal_action_probabilities)
                chosen_action = action_distribution.sample().item()

                log_prob_turn_total = log_prob_turn_total + torch.log(legal_action_probabilities[chosen_action])

                # Since chosen_action is an index of legal_action_probabilities, we need to map it back to the corresponding action
                chosen_action = legal_cells[chosen_action]


                # unit_action = torch.distributions.Categorical(probabilities).sample()

                if chosen_action < 6:
                    target_coords = self.game.map.get_neighbours_grid_coords(unit.r, unit.c)[chosen_action]
                    #
                    # if unit.name == 'Siege Sentinel':
                    #     print(f'{unit.r, unit.c} =====> {target_coords}')



                    # tried_to_go_on_own_unit = self.game.map.get(unit.r, unit.c).game_objects[0].player.nation == player.nation
                    # if target_coords is None or tried_to_go_on_own_unit:  # illegal move
                    #     player.reward += rewards_values.ILLEGAL_MOVE
                    #     break

                    if target_coords == unit_init_coords:
                        player.reward += rewards_values.RETURNED_TO_THE_INIT_POSITION
                else:
                    unit.path = []
                    # if unit.name == 'Siege Sentinel':
                    #     print(f'Siege Sentinel stays where he is')
                    # just stay where we are
                    break

                unit.move_one_cell(self.game, *target_coords)
                self.game.update()

        self.saved_log_probs.append(log_prob_turn_total)
