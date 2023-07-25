import json
import datetime
from typing import Tuple, List


class Event:
    def __init__(self, unit_type, unit_location, event_type):
        self.unit_type = unit_type
        self.unit_location = unit_location
        self.event_type = event_type

    def to_dict(self):
        return self.__dict__


class MoveEvent(Event):
    def __init__(self, unit_type: str, unit_location: Tuple[int, int], path: List[Tuple[int, int]]):
        super().__init__(unit_type, unit_location, 'move')
        self.path = path


class CombatAttackEvent(Event):
    def __init__(self, unit_type: str, unit_location: Tuple[int, int], target_location: Tuple[int, int],
                 unit_damage: int, enemy_damage: int):
        super().__init__(unit_type, unit_location, 'combat_attack')
        self.target_location = target_location
        self.unit_damage = unit_damage
        self.enemy_damage = enemy_damage


class RangedAttackEvent(Event):
    def __init__(self, unit_type: str, unit_location: Tuple[int, int], target_location: Tuple[int, int],
                 enemy_damage: int):
        super().__init__(unit_type, unit_location, 'ranged_attack')
        self.target_location = target_location
        self.enemy_damage = enemy_damage


class Logger:
    def __init__(self, map_size, map, initial_positions, log_path=None):
        self.initial_positions = initial_positions
        self.map = map
        self.map_size = map_size

        self.turns = []
        self.current_turn = None

        if log_path is not None:
            self.log_path = log_path
        else:
            self.log_path = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    def start_turn(self, turn_nation: str):
        self.current_turn = {
            "turnNation": turn_nation,
            "events": []
        }
        self.turns.append(self.current_turn)

    def log_event(self, event):
        event_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "eventType": event.event_type,
            "details": event.to_dict()
        }
        self.current_turn["events"].append(event_entry)

    def commit(self):
        with open(self.log_path, 'w') as f:
            f.write(self.to_json())

    def to_json(self):
        return json.dumps({
            'map_size': self.map_size,
            'map': self.map,
            'initial_positions': self.initial_positions,
            'turns': self.turns}, indent=2)

# import datetime
#
#
# def log(msg):
#     def wrapper1(func):
#         def wrapper2(*args, **kwargs):
#             result = func(*args, **kwargs)
#
#             formatted_args = kwargs
#             formatted_args.update({f'arg{i}': arg for i, arg in enumerate(args)})
#             formatted_args['result'] = result
#
#             with open('log.txt', 'a') as f:
#                 f.write(f'[{datetime.datetime.now()}] {msg.format(**formatted_args)}\n')
#             return result
#
#         return wrapper2
#
#     return wrapper1
#
#
# if __name__ == '__main__':
#     @log("Input values: {arg0}, {arg1}. Output: {result}")
#     def foo(a, b):
#         return a + b
#
#     foo(2, 3)
#     foo(3, 1)
#     foo(8, 4)
