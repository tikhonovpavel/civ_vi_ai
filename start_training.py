import pygame
import torch
import json
import os
import numpy as np

from tqdm import tqdm
import argparse

from matplotlib import pyplot as plt

from game import Game
from rl_training import QLearningAI
from rewards_values import Rewards


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 900


class TrainingSession:
    def __init__(self, silent=False, n_games=500, episode_max_length=5, replay_buffer_size=500):
        self.silent = silent
        self.n_games = n_games
        self.episode_max_length = episode_max_length
        self.replay_buffer_size = replay_buffer_size
        self.rewards = []

    def start_training(self):
        print(f'Start the training with the following parameters:')
        print(f'  Silent mode: {self.silent}')
        print(f'  Number of games: {self.n_games}')
        print(f'  Maximum episode length: {self.episode_max_length}')
        print(f'  Replay buffer size: {self.replay_buffer_size}')
        print('\n\n')

        if not self.silent:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            screen.fill((255, 255, 255))
            clock = pygame.time.Clock()
            clock.tick(60)
        else:
            screen = None

        n_games = self.n_games
        episode_max_length = self.episode_max_length

        with open('init_states/training_configs/1vs1_v2.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        start_game_index = 1#450

        models = None
        replay_buffer = None

        best_reward = float('-inf')
        
        for i in tqdm(range(start_game_index, n_games)):
            game = Game(config, screen, None, sound_on=False, autoplay=True, autoplay_max_turns=episode_max_length,
                        replay_buffer_size=self.replay_buffer_size, silent=self.silent)
            models, replay_buffer = game.players[0].ai.init(i, *(models, replay_buffer))
            queued_rewards = []

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

                # The most important lines:
                if current_player == rl_player:
                    current_player.create_paths(queued_rewards=queued_rewards)  
                    queued_rewards = []
                else:
                    current_player.create_paths()

                if game.turn_number > episode_max_length:
                    self.handle_game_end(rl_player, is_victory=False)
                    break

                if game.check_winning_conditions(current_player, no_units_eq_lose=True):
                    self.handle_game_end(rl_player, isinstance(current_player.ai, QLearningAI))
                    break

                for obj in current_player.game_objects:
                    res = obj.move(game, calc_rewards_for=[rl_player])[rl_player]
                    queued_rewards.extend(res)

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

            if game_reward > best_reward:
                best_reward = game_reward
                torch.save(models[0].state_dict(), f'weights/best_online_model_game_{str(i).zfill(5)}_score_{game_reward}.pt')
                torch.save(models[1].state_dict(), f'weights/best_reference_model_game_{str(i).zfill(5)}_score_{game_reward}.pt')

            if i % 100 == 0:
                torch.save(models[0].state_dict(), f'weights/online_model_game_{str(i).zfill(5)}.pt')
                torch.save(models[1].state_dict(), f'weights/reference_model_game_{str(i).zfill(5)}.pt')

                self.plot_rewards(f'rewards_history/game_{str(i).zfill(5)}.png')

            rl_player.ai.update_models()

        self.plot_rewards('rewards_history/final_result.png')

    def handle_game_end(self, rl_player, is_victory):
        reward_value = Rewards.get_named_reward(Rewards.VICTORY) if is_victory else Rewards.get_named_reward(Rewards.DEFEAT)
        print('QLearningAI won!' if is_victory else 'QLearningAI lost')

        if is_victory:
            print()
        
        for transition in rl_player.ai.replay_buffer.get_unfinished_transitions():
            rl_player.ai.replay_buffer.update_new_state_and_reward(turn_number=transition.turn_number,
                                                                   unit=transition.unit,
                                                                   new_state=None,
                                                                   additional_reward=reward_value,
                                                                   new_state_legal_action=None,)

        if not self.silent:
            print(f'Replay buffer final state:')
            print(rl_player.ai.replay_buffer)

    @staticmethod
    def running_average(data, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='valid')

    def plot_rewards(self, path):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, label='rewards')
        plt.plot(self.running_average(self.rewards, 50))
        
        plt.legend()
        plt.savefig(path)

    @staticmethod
    def cls():
        os.system('cls' if os.name=='nt' else 'clear')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start a training session.")
    parser.add_argument('--n_games', type=int, default=500, help="Number of games to play.")
    parser.add_argument('--replay_buffer_size', type=int, default=500, help="Size of the replay buffer.")
    parser.add_argument('--episode_max_length', type=int, default=5, help="Maximum length of an episode.")
    parser.add_argument('--silent', action='store_true', help="Run in silent mode without displaying the game.")
    args = parser.parse_args()

    TrainingSession.cls()
    print(args.silent)
    ts = TrainingSession(silent=args.silent, n_games=args.n_games,
                         episode_max_length=args.episode_max_length, replay_buffer_size=args.replay_buffer_size)
    ts.start_training()
