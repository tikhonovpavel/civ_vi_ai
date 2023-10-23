ENEMY_UNIT_DESTROYED = 150
OWN_UNIT_DESTROYED = -150

ENEMY_CITY_CAPTURED = 500
OWN_CITY_CAPTURED_BY_ENEMY = -500

VICTORY = 1000
DEFEAT = -1000
DRAW = -5


_ENEMY_UNIT_DAMAGED_COEFF = 1
_OWN_UNIT_DAMAGED_COEFF = 1


def enemy_unit_damaged_reward(damage):
    return _ENEMY_UNIT_DAMAGED_COEFF * damage


def own_unit_damaged_reward(damage):
    return -_OWN_UNIT_DAMAGED_COEFF * damage


# ILLEGAL_MOVE = -0.1
# RETURNED_TO_THE_INIT_POSITION = -0.1
