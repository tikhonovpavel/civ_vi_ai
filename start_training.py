import json
import os
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt

from game import Game
from rl_training import QLearningAI
from rewards_values import Rewards

from line_profiler_pycharm import profile
import pygame


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 900

class TrainingSession:
    def __init__(self, silent=False, n_games=500, episode_max_length=10):
        self.silent = silent
        self.n_games = n_games
        self.episode_max_length = episode_max_length
        self.rewards = []

    @profile
    def start_training(self):

        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.fill((255, 255, 255))
        clock = pygame.time.Clock()
        clock.tick(60)

        n_games = self.n_games
        episode_max_length = self.episode_max_length

        with open('init_states/training_configs/1vs1_very_easy.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        models, replay_buffer = None, None
        for i in tqdm(range(1, n_games)):
            game = Game(config, screen, None, sound_on=False,
                    autoplay=True, autoplay_max_turns=episode_max_length, silent=self.silent)
            models, replay_buffer = game.players[0].ai.init(i, *(models, replay_buffer))

            rl_player = game.players[0]

            game.start()

            game.turn_number = 1
            game.subturn_number = 0
            while True:
                # continue
                current_player = game.get_current_player()

                if not self.silent:
                    print(f'\n\n== [Game {i}] {current_player.nation} started the turn {game.turn_number} '
                        f'with {len(current_player.units)} units and {len(current_player.cities)} cities: ==')

                game.logger.start_turn(current_player.nation)

                # The most important line:
                current_player.create_paths()  

                if game.turn_number > 20:
                    self.handle_game_end(rl_player, is_victory=False)
                    break

                if game.check_winning_conditions(current_player, no_units_eq_lose=True):
                    self.handle_game_end(rl_player, isinstance(current_player.ai, QLearningAI))
                    break

                for obj in current_player.game_objects:
                    if current_player.nation == 'Rome':
                        print('hoba')

                    res = obj.move(game, calc_rewards_for=[rl_player])#[0]
                    print(f'res of the move: {res}')
                    obj.gain_hps()

                    obj.mp = obj.mp_base
                    obj.can_attack = True
                    obj.is_selected = False

                if game.check_winning_conditions(current_player, no_units_eq_lose=True):
                    self.handle_game_end(rl_player, isinstance(current_player.ai, QLearningAI))
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
            self.rewards.append(game_reward)

            rl_player.ai.update_models()

        self.plot_rewards()

    def handle_game_end(self, rl_player, is_victory):
        reward_value = Rewards.get_named_reward(Rewards.VICTORY) if is_victory else Rewards.get_named_reward(Rewards.DEFEAT)
        print('QLearningAI won!' if is_victory else 'QLearningAI lost')
        
        for transition in rl_player.ai.replay_buffer.get_unfinished_transitions():
            rl_player.ai.replay_buffer.update_new_state_and_reward(turn_number=transition.turn_number,
                                                                   unit=transition.unit,
                                                                   new_state=None,
                                                                   additional_reward=reward_value,
                                                                   new_state_legal_action=None,)
            
        if not self.silent:
            print(f'Replay buffer final state:')
            print(rl_player.ai.replay_buffer)

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, label='rewards')
        plt.legend()
        plt.savefig('rewards.png')

    @staticmethod
    def cls():
        os.system('cls' if os.name=='nt' else 'clear')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start a training session.")
    parser.add_argument('--n_games', type=int, default=500, help="Number of games to play.")
    parser.add_argument('--episode_max_length', type=int, default=10, help="Maximum length of an episode.")
    parser.add_argument('--silent', action='store_true', help="Run in silent mode without displaying the game.")
    args = parser.parse_args()

    TrainingSession.cls()
    print(args.silent)
    ts = TrainingSession(silent=args.silent, n_games=args.n_games, episode_max_length=args.episode_max_length)
    ts.start_training()
