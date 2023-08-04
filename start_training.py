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
    n_games = 1000
    episode_max_length = 10

    rewards = []

    with open('init_states/1vs1_easy2.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    for i in tqdm(range(n_games)):
        game = Game(config, None, None, sound_on=False,
                    autoplay=True, autoplay_max_turns=episode_max_length)
        try:
            game.start()
        except RuntimeError as err:
            if 'stack expects a non-empty TensorList' not in err.args:
                raise err

        rl_player = next(p for p in game.players if isinstance(p.ai, PolicyGradientAI))
        print(f'Reward of the game {i + 1}/{n_games} is: {rl_player.reward_cum}')

        rewards.append(rl_player.reward_cum)

        rl_player.ai.update_policy()

    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    start_training()
