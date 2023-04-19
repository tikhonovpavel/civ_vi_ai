from line_profiler_pycharm import profile
import pygame

# Define colors
from grid import TerrainTypes

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

HEX_DEFAULT = (0, 255, 0)
HEX_SELECTED = (0, 128, 0)

# Set the size for the image
DEFAULT_UNIT_IMAGE_SIZE = (30, 30)
DEFAULT_TERRAIN_IMAGE_SIZE = (40, 37)
UI_UNIT_IMAGE_SIZE = (100, 100)

DEFAULT_NATION_IMAGE_SIZE = (15, 10)
UI_NATION_IMAGE_SIZE = (50, 50)


pygame.font.init()


class Image:
    def __init__(self):
        self.nations = ...
        self.units = ...
        self.tiles = {k.image_path: pygame.image.load(k.image_path) for k in [TerrainTypes.FOREST,
                                                                              TerrainTypes.HILLS,
                                                                              TerrainTypes.PLAINS]}
    #
    # image = pygame.image.load(image_path)
    # self.image = pygame.transform.scale(image, DEFAULT_UNIT_IMAGE_SIZE)
    # self.image_ui = pygame.transform.scale(image, UI_UNIT_IMAGE_SIZE)


class Text:
    def __init__(self):
        self._fonts = dict()
        self._texts = dict()

    @profile
    def _get_font(self, font_family, size):
        if font_family not in self._fonts:
            self._fonts[font_family] = {}

        if size not in self._fonts[font_family]:
            self._fonts[font_family][size] = pygame.font.SysFont(font_family, size)

        return self._fonts[font_family][size]

    @profile
    def text_to_screen(self, screen, text, x, y, size=50, color=(200, 000, 000), font_family='Arial'):
        text = str(text)

        if text not in self._texts:
            self._texts[text] = self._get_font(font_family, size).render(text, True, color)

        screen.blit(self._texts[text], (x, y))
@profile
def get_nation_icon(nation):
    return f'assets/nations/{nation}.png'


nation_icons = dict()
for nation in ['rome', 'egypt']:
    image = pygame.image.load(get_nation_icon(nation))
    nation_icons[nation] = {'default': pygame.transform.scale(image, DEFAULT_NATION_IMAGE_SIZE),
                            'ui': pygame.transform.scale(image, UI_NATION_IMAGE_SIZE)}


class Display:
    def __init__(self, screen, game) -> None:
        self.screen = screen
        self.game = game
        self.text_module = Text()
        self.images = Image()



    @profile
    def update_all(self):  # players, hexagon_grid, paths):
        self.screen.fill((255, 255, 255))

        self._update_grid()
        self._update_units()
        self._update_paths()
        self._update_ui()

        pygame.display.update()

    @profile
    def _update_grid(self, ):
        for r, row in enumerate(self.game.map.tiles):
            for c, hex in enumerate(row):
                terrain_image = self.images.tiles[hex.terrain.image_path]
                terrain_image = pygame.transform.scale(terrain_image, DEFAULT_TERRAIN_IMAGE_SIZE)
                self.screen.blit(terrain_image,
                                 (hex.x - DEFAULT_TERRAIN_IMAGE_SIZE[0] / 2,
                                  hex.y - DEFAULT_TERRAIN_IMAGE_SIZE[1] / 2))
                # pygame.draw.polygon(self.screen, HEX_DEFAULT, hex.points, 0)
                # pygame.draw.polygon(self.screen, BLACK, hex.points, 1)


                # self.text_module.text_to_screen(self.screen, f'{(r, c)}', hex.x - 12, hex.y - 3, size=10)

        for r, row in enumerate(self.game.map.tiles):
            for c, hex in enumerate(row):
                pygame.draw.polygon(self.screen, BLACK, hex.points, 1)

    def _update_units(self, ):

        for player in self.game.players:
            # player.flag

            for unit in player.units:
                unit_hex = self.game.map.get(unit.r, unit.c)

                if unit.is_selected:
                    pygame.draw.polygon(self.screen, HEX_SELECTED, unit_hex.points, 0)
                    pygame.draw.polygon(self.screen, BLACK, unit_hex.points, 1)

                nation_image = nation_icons[player.nation]['default']
                self.screen.blit(unit.image, (unit_hex.x - unit.image.get_width() / 2, unit_hex.y - unit.image.get_height() / 2))
                self.screen.blit(nation_image, (unit_hex.x - nation_image.get_width() / 2, unit_hex.y + 7))

                hp_offset = 15
                hp_length = 16
                hp_thickness = 5

                pygame.draw.rect(
                    self.screen,
                    (255, 0, 0),
                    (unit_hex.x - hp_length / 2, unit_hex.y - hp_offset, hp_length * (unit.hp / 100), hp_thickness),)

                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),
                    (unit_hex.x - hp_length / 2, unit_hex.y - hp_offset, hp_length, hp_thickness),
                    width=1)

    def _update_paths(self):
        unit_selected = self.game.get_selected_unit()
        if unit_selected is None:
            return

        for i, (path_r, path_c) in enumerate(unit_selected.path):
            radius = 8 if i == 0 else 5

            path_hex = self.game.map.get(path_r, path_c)
            pygame.draw.circle(self.screen, (128, 0, 128), (path_hex.x, path_hex.y), radius=radius)

            self.text_module.text_to_screen(self.screen, f'{i}', x=path_hex.x, y=path_hex.y, size=20)

    def _update_ui(self):
        units = self.game.get_current_player().units

        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (50, 600, 500, 150),
            width=0)

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (50, 600, 500, 150),
            width=2)

        unit = next((u for u in units if u.is_selected), None)
        if unit is not None:
            self.screen.blit(unit.image_ui, (55, 605))
            self.text_module.text_to_screen(self.screen,
                                            f'HP: {unit.hp}',
                                            x=55,  # +UI_UNIT_IMAGE_SIZE[0]+5,
                                            y=605 + UI_UNIT_IMAGE_SIZE[1] + 5,
                                            size=30)

            enemy_unit = self.game.get_current_player().enemy_unit
            if enemy_unit is None:
                return

            self.screen.blit(enemy_unit.image_ui, (405, 605))
            self.text_module.text_to_screen(self.screen,
                                            f'HP: {enemy_unit.hp}',
                                            x=405,  # +UI_UNIT_IMAGE_SIZE[0]+5,
                                            y=605 + UI_UNIT_IMAGE_SIZE[1] + 5,
                                            size=30)
