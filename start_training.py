import json

import torch
from line_profiler_pycharm import profile
from matplotlib import pyplot as plt
from tqdm import tqdm

from game import Game
from rl_training import PolicyGradientAI


@profile
def start_training():
    #     torch.manual_seed(42)
    n_games = 10
    episode_max_length = 10

    rewards = []
    rewards_lengths = []

    with open('init_states/training_configs/1vs1_easy.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    policy_network = None
    for i in tqdm(range(n_games)):
        game = Game(config, None, None, sound_on=False,
                    autoplay=True, autoplay_max_turns=episode_max_length)
        policy_network = game.players[1].ai.init_policy_network(policy_network)

        try:
            game.start()
        except RuntimeError as err:
            if 'stack expects a non-empty TensorList' not in err.args:
                raise err

        rl_player = next(p for p in game.players if isinstance(p.ai, PolicyGradientAI))
        print(f'Reward of the game {i + 1}/{n_games} is: {rl_player.reward_cum}')

        rl_player.ai.update_policy()

        rewards.append(rl_player.reward_cum)
        rewards_lengths.append(len(rl_player.ai._rewards_history))

    plt.plot(rewards, label='rewards')
    plt.plot(rewards_lengths, label='rewards length')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    start_training()
