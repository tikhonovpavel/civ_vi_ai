class City:

    def __init__(self, name, center_r, center_c):
        self.name = name
        self.center_r = center_r
        self.center_c = center_c

        self.hp = 200
        self.walls_hp = 100

        self.tiles_set = {(center_r, center_c),}

    def is_cell_inside(self, r, c):
        return next((True for tile in self.tiles_set if tile == (r, c)), False)