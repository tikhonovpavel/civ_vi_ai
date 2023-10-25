from enum import Enum


class Rewards(Enum):


    ENEMY_UNIT_DESTROYED = 200
    OWN_UNIT_DESTROYED = -150
    
    ENEMY_CITY_CAPTURED = 500
    OWN_CITY_CAPTURED_BY_ENEMY = -500

    SURVIVED_THE_TURN = 5
    STAYING_WHERE_HE_IS = -1
    
    VICTORY = 1000
    DEFEAT = -1000
    DRAW = -100

    ENEMY_UNIT_DAMAGED = "ENEMY_UNIT_DAMAGED"
    OWN_UNIT_DAMAGED = "OWN_UNIT_DAMAGED"

    _ENEMY_UNIT_DAMAGED_COEFF = 1.5
    _OWN_UNIT_DAMAGED_COEFF = -1

    @classmethod
    def get_named_reward(cls, reward_name, damage=0, to_unit=None):
        if reward_name == cls.ENEMY_UNIT_DAMAGED:
            return {reward_name.name: int(damage * cls._ENEMY_UNIT_DAMAGED_COEFF.value), 'to_unit': to_unit}
        elif reward_name == cls.OWN_UNIT_DAMAGED:
            return {reward_name.name: int(damage * cls._OWN_UNIT_DAMAGED_COEFF.value), 'to_unit': to_unit}
        elif reward_name in cls:
            return {reward_name.name: reward_name.value, 'to_unit': to_unit}
        return None


if __name__ == '__main__':
    print(Rewards.get_named_reward(Rewards.VICTORY))
    print(Rewards.get_named_reward(Rewards.ENEMY_UNIT_DAMAGED, 37))
    print(Rewards.get_named_reward(Rewards.OWN_UNIT_DAMAGED, 10))
