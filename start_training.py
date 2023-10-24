import json
import os
from tqdm import tqdm

from line_profiler_pycharm import profile

from matplotlib import pyplot as plt

from game import Game
from rl_training import QLearningAI
from rewards_values import Rewards

@profile
def start_training():
    n_games = 100
    episode_max_length = 10

    rewards = []
    with open('init_states/training_configs/1vs1_easy.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    models, replay_buffer = None, None
    for i in tqdm(range(1, n_games)):
        game = Game(config, None, None, sound_on=False,
                    autoplay=True, autoplay_max_turns=episode_max_length)
        models, replay_buffer = game.players[0].ai.init(i, *(models, replay_buffer))

        rl_player = game.players[0]

        game.start()

        game.turn_number = 1
        game.subturn_number = 0
        while True:
            current_player = game.get_current_player()
            print(f'\n\n== [Game {i + 1}] {current_player.nation} started the turn {game.turn_number} '
                  f'with {len(current_player.units)} units and {len(current_player.cities)} cities: ==')

            game.logger.start_turn(current_player.nation)

            current_player.create_paths()

            if game.turn_number > 10:
                handle_game_end(rl_player, is_victory=False)
                break

            if game.check_winning_conditions(current_player, no_units_eq_lose=True):
                handle_game_end(rl_player, isinstance(current_player.ai, QLearningAI))
                break

            for obj in current_player.game_objects:
                obj.move(game)
                obj.gain_hps()

                obj.mp = obj.mp_base
                obj.can_attack = True
                obj.is_selected = False

            if game.check_winning_conditions(current_player, no_units_eq_lose=True):
                handle_game_end(rl_player, isinstance(current_player.ai, QLearningAI))
                break

            # ----------------------------------
            # next player
            game.set_next_current_player()
            game.update()

            game.logger.commit()

            game.current_turn_text.update(turn_number=game.turn_number)
            game.current_player_text.update(current_player=current_player.nation)
            game.update()

            game.subturn_number += 1
            if game.subturn_number % len(game.players) == 0:
                game.turn_number += 1

        game_reward = rl_player.ai.replay_buffer.get_last_game_total_reward()
        print(f'At the end of the game {i}, the rewards: {game_reward}')
        rewards.append(game_reward)
        
        rl_player.ai.update_models()

    plt.figure(figsize=(10, 10))
    plt.plot(rewards, label='rewards')
    plt.legend()
    plt.savefig('rewards.png')

def handle_game_end(rl_player, is_victory):
    reward_value = Rewards.get_named_reward(Rewards.VICTORY) if is_victory else Rewards.get_named_reward(Rewards.DEFEAT)
    print('QLearningAI won!' if is_victory else 'QLearningAI lost')
    
    for transition in rl_player.ai.replay_buffer.get_unfinished_transitions():
        rl_player.ai.replay_buffer.update_new_state_and_reward(
            turn_number=transition.turn_number,
            unit=transition.unit,
            new_state=None,
            additional_reward=reward_value,
            new_state_legal_action=None,)
        
    print(f'Replay buffer final state:')
    print(rl_player.ai.replay_buffer)

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

if __name__ == '__main__':
    cls()
    start_training()
