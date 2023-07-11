import pygame


class City:
    DEFAULT_CITY_IMAGE_SIZE = (42, 42)

    MAX_HP = 200
    MAX_WALLS_HP = 100

    def __init__(self, name, center_r, center_c, tiles_set, image_path):
        assert center_r, center_c in tiles_set

        self.name = name
        self.center_r = center_r
        self.center_c = center_c

        self.image_path = image_path

        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, self.DEFAULT_CITY_IMAGE_SIZE)

        self.hp = City.MAX_HP
        self.walls_hp = City.MAX_WALLS_HP

        self._tiles_set = tiles_set

    @property
    def tiles_set(self):
        return self._tiles_set

    @tiles_set.setter
    def tiles_set(self, tiles_set):
        self._tiles_set = tiles_set

    def is_cell_inside(self, r, c):
        return next((True for tile in self._tiles_set if tile == (r, c)), False)

    def draw(self, screen, game, text_module):
        center_hex = game.map.get(self.center_r, self.center_c).hex

        screen.blit(self.image, (center_hex.x - self.DEFAULT_CITY_IMAGE_SIZE[0] / 2,
                                 center_hex.y - self.DEFAULT_CITY_IMAGE_SIZE[1] / 2))

        hp_offset = 20
        hp_length = 32
        hp_thickness = 8

        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (center_hex.x - hp_length / 2, center_hex.y - hp_offset, hp_length * (self.hp / self.MAX_HP), hp_thickness))

        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (center_hex.x - hp_length / 2, center_hex.y - hp_offset, hp_length, hp_thickness),
            width=1)

        text_module.text_to_screen(screen, self.name, center_hex.x, center_hex.y + 15,
                                   size=15, color=(0, 0, 0), bold=True, align='center')



