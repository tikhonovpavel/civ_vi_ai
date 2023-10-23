import json

import torch
from line_profiler_pycharm import profile
from matplotlib import pyplot as plt
from tqdm import tqdm

from game import Game
from rl_training import QLearningAI


@profile
def start_training():
    n_games = 10
    episode_max_length = 10

    rewards = []
    rewards_lengths = []

    with open('init_states/training_configs/1vs1_easy.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    models = None
    for i in tqdm(range(n_games)):
        game = Game(config, None, None, sound_on=False,
                    autoplay=True, autoplay_max_turns=episode_max_length)
        models = game.players[0].ai.init_models(models, i)

        rl_player = game.players[0]

        game.start()

        game.turn_number = 1
        game.subturn_number = 0
        while True:
            current_player = game.get_current_player()
            print(f'\n\n== {current_player.nation} started the turn {game.turn_number} '
                  f'with {len(current_player.units)} units and {len(current_player.cities)} cities: ==')

            game.logger.start_turn(current_player.nation)

            current_player.create_paths()

            if game.check_winning_conditions(current_player, no_units_eq_lose=True):
                # ну собственно надо здесь сделать апдейт состояний и ревордов
                if isinstance(current_player.ai, QLearningAI):
                    print('QLearningAI won!')
                else:
                    print('QLearningAI lost')
                break

            for obj in current_player.game_objects:
                obj.move(game)
                obj.gain_hps()

                obj.mp = obj.mp_base
                obj.can_attack = True
                obj.is_selected = False

            if game.check_winning_conditions(current_player, no_units_eq_lose=True):
                # ну собственно надо здесь сделать апдейт состояний и ревордов x2
                if isinstance(current_player.ai, QLearningAI):
                    print('QLearningAI won!')
                else:
                    print('QLearningAI lost')
                # reward += 1000
                break

            # if isinstance(current_player, TrainableAI):
            #     current_player.receive_reward(reward)

            # ----------------------------------
            # next player
            game.set_next_current_player()
            game.update()

            game.logger.commit()

            game.current_turn_text.update(turn_number=game.turn_number)
            game.current_player_text.update(current_player=current_player.nation)
            game.update()

            game.subturn_number += 1
            # print(f'SUBturn number switched from {game.subturn_number - 1} to {game.subturn_number}')
            if game.subturn_number % len(game.players) == 0:
                game.turn_number += 1
                # print(f'Turn number switched from {game.turn_number - 1} to {game.turn_number}')

        # rl_player = next(p for p in game.players if isinstance(p.ai, QLearningAI))
        print(f'Reward of the game {i + 1}/{n_games} is: {rl_player.reward_cum}')

        rl_player.ai.update_models()

        rewards.append(rl_player.reward_cum)
        rewards_lengths.append(len(rl_player.ai._rewards_history))

    plt.plot(rewards, label='rewards')
    plt.plot(rewards_lengths, label='rewards length')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    start_training()
