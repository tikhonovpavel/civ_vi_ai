import numbers
import random
from dataclasses import dataclass
from typing import List, Dict


import torch
from tabulate import tabulate

from unit import Unit


random.seed(42)
torch.manual_seed(42)


@dataclass
class Transition:
    game_number: int
    turn_number: int
    movement_number: int  # number of the move of the given unit inside one turn
    unit: Unit
    s: torch.Tensor
    a: int
    r: List[Dict[str, int]]  # List of dictionaries to store reasons for rewards
    s_next: torch.Tensor | None
    legal_actions_s_next: List[int] | None

    @property
    def total_reward(self):
        return sum([event[key] for event in self.r for key in event
                    if isinstance(event[key], numbers.Number)])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []

    def filter(self, unit_name):
        return [rb for rb in self.buffer if rb.unit.name == unit_name]

    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def update_new_state_and_reward(self, turn_number, unit, new_state, new_state_legal_action, additional_reward: Dict[str, int]):
        """
        Updates the state and reward for the specified unit and turn.

        Parameters:
        - turn_number (int): The turn number for which the information needs to be updated.
        - unit (Unit): The unit for which the information needs to be updated.
        - new_state (Tensor): The new state to set.
        - additional_reward (Dict[str, int]): Dictionary containing the reason and the amount of the additional reward.

        Raises an exception if:
        - No corresponding transition for update is found.
        - More than one candidate for update is found.
        """

        if not isinstance(additional_reward, list):
            raise Exception()

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
                transition.r.extend(additional_reward)  # Add the additional reward to the list
                updated = True
                break

        if not updated:
            raise ValueError(f"No transition found for unit {unit} on turn {turn_number} with s_next as None.")
        
    def get_last_game_total_reward(self):
        last_game_number = self.buffer[-1].game_number
        return sum(transition.total_reward for transition in self.buffer if transition.game_number == last_game_number)

    def get_unfinished_transitions(self):
        """
        Returns transitions for which the reward has not been determined yet.
        Raises an exception if these transitions are not from the last game or the last turn.
        """
        # Check that the buffer is not empty
        if not self.buffer:
            raise Exception('Called on the empty buffer')

        # Get the number of the last game and the last turn from the last transition in the buffer
        last_game_number = self.buffer[-1].game_number
        last_turn_number = self.buffer[-1].turn_number

        unfinished_transitions = []

        for transition in reversed(self.buffer):
            # If the reward for the transition is not defined, add it to the list
            if transition.r is None:
                unfinished_transitions.append(transition)

                # If this transition doesn't belong to the last turn or the last game, raise an exception
                if transition.game_number != last_game_number or transition.turn_number != last_turn_number:
                    raise ValueError(f"Unfinished transition found for game {transition.game_number} "
                                     f"and turn {transition.turn_number}, which is not the last game or turn.")

        return unfinished_transitions

    def __str__(self):
        headers = ["Game", "Turn", "Move", "Unit", "Action", "RewardNames", "RewardValues", "State", "State Next"]
        table_data = []

        for transition in self.buffer:
            row = [
                transition.game_number,
                transition.turn_number,
                transition.movement_number,
                str(transition.unit),
                transition.a,
                ' + '.join(next(k for k, v in r.items() if isinstance(v, numbers.Number)) for r in transition.r),
                ''.join(next((f'+{v}' if v >= 0 else str(v)) for v in r.values() if isinstance(v, numbers.Number)) for r in transition.r) + f'={transition.total_reward}',
                "None" if transition.s is None else "+",
                "None" if transition.s_next is None else "+"
            ]
            table_data.append(row)

        return tabulate(table_data, headers=headers)
