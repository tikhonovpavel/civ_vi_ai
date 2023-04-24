from datetime import datetime

from line_profiler_pycharm import profile
import pygame

# Define colors
from button import ButtonStates
from map import TerrainTypes

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set the size for the image
DEFAULT_UNIT_IMAGE_SIZE = (30, 30)
DEFAULT_TERRAIN_IMAGE_SIZE = (40, 37)
UI_UNIT_IMAGE_SIZE = (100, 100)

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
    def _get_font(self, font_family='Arial', size=50, bold=False, alpha=255):
        setting = (font_family, size, bold)

        if setting not in self._fonts:
            self._fonts[setting] = pygame.font.SysFont(font_family, size, bold=bold)

        return self._fonts[setting]

    @profile
    def text_to_screen(self, screen, text, x, y, size=50, bold=False, color=(200, 000, 000),
                       font_family='Arial', alpha=255):
        text = str(text)

        if text not in self._texts:
            font = self._get_font(font_family, size, bold, alpha)

            # font.set_alpha(alpha)
            self._texts[text] = font.render(text, True, color)

            # arr = pygame.surfarray.pixels_alpha(self._texts[text])
            # arr[:, :] = (arr[:, :] * (50 / 255)).astype(arr.dtype)

        screen.blit(self._texts[text], (x, y))





class Display:

    def __init__(self, screen, game) -> None:
        self.screen = screen
        self.game = game
        self.text_module = Text()
        self.images = Image()

        self.damage_texts = []

    class DisappearingText:
        def __init__(self, game, text, r, c):
            self.start_time = datetime.now()
            self.text = text

            hex = game.map.get(r, c).hex
            self.x, self.y = hex.x, hex.y

    def show_damage_text(self, text, r, c):
        self.damage_texts.append(self.DisappearingText(self.game, text, r, c))

    @profile
    def update_all(self):
        self.screen.fill((255, 255, 255))

        self._update_grid()
        self._update_units()
        self._update_paths()
        self._update_ui()

        self._print_grid_coords()

        pygame.display.update()

    def _print_grid_coords(self):
        for r, row in enumerate(self.game.map.tiles):
            for c, hex in enumerate(row):
                self.text_module.text_to_screen(self.screen, f'{(r, c)}', hex.x - 12, hex.y - 3,
                                                size=10, color=(0,0,0))
    @profile
    def _update_grid(self, ):
        for r, row in enumerate(self.game.map.tiles):
            for c, tile in enumerate(row):
                terrain_image = self.images.tiles[tile.terrain.image_path]
                terrain_image = pygame.transform.scale(terrain_image, DEFAULT_TERRAIN_IMAGE_SIZE)
                self.screen.blit(terrain_image,
                                 (tile.x - DEFAULT_TERRAIN_IMAGE_SIZE[0] / 2,
                                  tile.y - DEFAULT_TERRAIN_IMAGE_SIZE[1] / 2))

        for r, row in enumerate(self.game.map.tiles):
            for c, tile in enumerate(row):
                pygame.draw.polygon(self.screen, BLACK, tile.points, 1)

    def _update_units(self):
        for player in self.game.players:
            for unit in player.units:
                unit.draw(self.screen, self.game)

    def _update_paths(self):
        unit_selected = self.game.get_selected_unit()
        if unit_selected is None or len(unit_selected.path) == 0:
            return

        # mp_used = 0
        mp_left = unit_selected.mp
        last_step_number = 0
        path_r_prev, path_c_prev = unit_selected.path[0]

        indices = []
        for i, (path_r, path_c) in enumerate(unit_selected.path[1:]):

            move_cost = self.game.map.get_data_edge((path_r_prev, path_c_prev), (path_r, path_c))['cost']

            # print(f'Move cost from {(path_r_prev, path_c_prev)} to {(path_r, path_c)} is {move_cost}. MP left: {mp_left}')
            if mp_left < move_cost:
                step_number = last_step_number + 1
                mp_left = unit_selected.mp_base
                mp_left -= move_cost
                indices.append(i-1)
            else:
                step_number = last_step_number
                mp_left -= move_cost

            # print(f'=> step={step_number}. And now MP left if {mp_left}')

            # print()


            last_step_number = step_number
            path_r_prev, path_c_prev = path_r, path_c

        for i, (path_r, path_c) in enumerate(unit_selected.path[1:]):
            path_hex = self.game.map.get(path_r, path_c).hex

            if i in indices or (i == len(unit_selected.path) - 2):
                radius = 8
                pygame.draw.circle(self.screen, (128, 0, 128), (path_hex.x, path_hex.y), radius=radius)

                self.text_module.text_to_screen(self.screen,
                                                f'{(indices.index(i) if i in indices else len(indices)) + 1}',
                                                x=path_hex.x, y=path_hex.y,
                                                size=25, bold=True, color=(255, 255, 255))
            else:
                radius = 4
                pygame.draw.circle(self.screen, (128, 0, 128), (path_hex.x, path_hex.y), radius=radius)



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
                                            x=55,
                                            y=605 + UI_UNIT_IMAGE_SIZE[1] + 5,
                                            size=30)

            enemy_unit = self.game.get_current_player().enemy_unit
            if enemy_unit is None:
                return

            self.screen.blit(enemy_unit.image_ui, (405, 605))
            self.text_module.text_to_screen(self.screen,
                                            f'HP: {enemy_unit.hp}',
                                            x=405,
                                            y=605 + UI_UNIT_IMAGE_SIZE[1] + 5,
                                            size=30)

        self.game.next_move_button.draw(self.screen, self.game, self.text_module)

    def update_texts(self):
        to_delete = []

        for i in range(len(self.damage_texts)):
            damage_text = self.damage_texts[i]

            if (datetime.now() - damage_text.start_time).total_seconds() >= 1.5:
                to_delete.append(i)
                continue

            self.text_module.text_to_screen(self.screen,
                                            damage_text.text,
                                            x=damage_text.x,
                                            y=damage_text.y,
                                            bold=True,
                                            size=28,
                                            color=(255, 255, 255))

        pygame.display.flip()

        for index in sorted(to_delete, reverse=True):
            try:
                self.damage_texts.pop(index)
                self.update_all()
            except:
                pass