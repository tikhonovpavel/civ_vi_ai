import json
import datetime
from typing import Tuple, List

from game_object import MilitaryObject


class Event:
    def __init__(self, obj, event_type):
        self.category = obj.category
        self.location = (obj.r, obj.c)
        self.hp = obj.hp
        self.mp = obj.mp
        self.name = obj.name
        self.event_type = event_type

    def to_dict(self):
        return self.__dict__


class MoveEvent(Event):
    def __init__(self, obj: MilitaryObject, path: List[Tuple[int, int]]):
        super().__init__(obj, 'move')
        self.path = path


class CombatAttackEvent(Event):
    def __init__(self, obj: MilitaryObject, target: MilitaryObject,
                 unit_damage: int, enemy_damage: int):
        super().__init__(obj, 'combat_attack')
        self.target = {'target': {'name': target.name, 'location': (target.r, target.c), 'category': target.category}}
        self.unit_damage = unit_damage
        self.enemy_damage = enemy_damage


class RangedAttackEvent(Event):
    def __init__(self, obj: MilitaryObject, target: MilitaryObject, enemy_damage: int):
        super().__init__(obj, 'ranged_attack')
        self.target = {'name': target.name, 'location': (target.r, target.c), 'category': target.category}
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
            'turns': self.turns,
            'map_size': self.map_size,
            'map': self.map,
            'initial_positions': self.initial_positions}, indent=2)

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
