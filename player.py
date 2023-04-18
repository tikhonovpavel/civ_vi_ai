class Player:

    def __init__(self, nation, units) -> None:
        self.nation = nation
        self.units = units

        self.ready_to_attack_player = None
        self.ready_to_attack_unit = None
        # self.ready_to_attack_hex = (None, None)


    def set_ready_to_attack(self, player, unit):
        self.ready_to_attack_player = player
        self.ready_to_attack_unit = unit
        # self.ready_to_attack_hex = (hex_r, hex_c)


    def no_attack(self,):
        self.ready_to_attack_player = None
        self.ready_to_attack_unit = None
        # self.ready_to_attack_hex = (None, None)
