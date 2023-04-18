class Player:

    def __init__(self, nation, units) -> None:
        self.nation = nation
        self.units = units

        self.enemy_player = None
        self.enemy_unit = None
        # self.ready_to_attack_hex = (None, None)


    def set_enemy(self, player, unit):
        self.enemy_player = player
        self.enemy_unit = unit
        # self.ready_to_attack_hex = (hex_r, hex_c)

    def get_enemy(self):
        return self.enemy_player, self.enemy_unit


    def no_attack(self,):
        self.enemy_player = None
        self.enemy_unit = None
        # self.ready_to_attack_hex = (None, None)
