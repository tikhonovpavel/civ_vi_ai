from enum import Enum


class Rewards(Enum):
    
    
    ENEMY_UNIT_DESTROYED = 200
    OWN_UNIT_DESTROYED = -150
    
    ENEMY_CITY_CAPTURED = 500
    OWN_CITY_CAPTURED_BY_ENEMY = -500

    SURVIVED_THE_TURN = 0
    STAYING_WHERE_HE_IS = 0
    
    VICTORY = 1000
    DEFEAT = -1000
    DRAW = -5

    @classmethod
    def get_named_reward(cls, reward_name):
        if reward_name in cls:
            return {reward_name.name: reward_name.value}
        return None

if __name__ == '__main__':
    print(Rewards.get_named_reward(Rewards.VICTORY))  # {'VICTORY': 1000}
