from datetime import datetime

# from line_profiler_pycharm import profile
import pygame

# Define colors
# import game
from city import City
from consts import DEFAULT_TERRAIN_IMAGE_SIZE, BLACK, UI_UNIT_IMAGE_SIZE
from map import terrain_types
from nations import Nations
from ui import ButtonStates
# from map import TerrainTypes

from line_profiler_pycharm import profile

pygame.font.init()


class Image:
    def __init__(self):
        self.nations = ...
        self.units = ...
        self.tiles = {k.image_path: pygame.image.load(k.image_path) for k in terrain_types.values()}


class Text:
    def __init__(self):
        self._fonts = dict()
        self._texts = dict()

    # @profile
    def _get_font(self, font_family='Arial', size=50, bold=False, alpha=255):
        setting = (font_family, size, bold)

        if setting not in self._fonts:
            self._fonts[setting] = pygame.font.SysFont(font_family, size, bold=bold)

        return self._fonts[setting]

    def _text_to_screen(self, screen, text, x, y, size=50, bold=False, color=(200, 000, 000),
                       font_family='Arial', alpha=255, align='left'):
        text = str(text)

        if text not in self._texts:
            font = self._get_font(font_family, size, bold, alpha)

            # font.set_alpha(alpha)
            self._texts[text] = font.render(text, True, color)

        if align == 'left':
            screen.blit(self._texts[text], (x, y))
        elif align == 'center':
            text_width = self._texts[text].get_rect().width
            screen.blit(self._texts[text], (x - text_width // 2, y))
        else:
            raise NotImplementedError()

    # @profile
    def text_to_screen(self, screen, text, x, y, size=50, bold=False, color=(200, 000, 000),
                       font_family='Arial', alpha=255, align='left'):
        for i, t in enumerate(text.split('\n')):
            self._text_to_screen(screen, t, x, y + i * size, size, bold, color, font_family, alpha, align)



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

            geom = game.map.get(r, c).geometry
            self.x, self.y = geom.x, geom.y

    def show_damage_text(self, text, r, c):
        self.damage_texts.append(self.DisappearingText(self.game, text, r, c))

    @profile
    def update_all(self):
        if self.screen is None:
            return

        self.screen.fill((255, 255, 255))
        
        self._update_grid()
        self._update_units_and_cities()
        self._update_paths()
        self._update_ui()
        
        self._print_grid_coords()
        
        pygame.display.update()

    def _print_grid_coords(self):
        for r, row in enumerate(self.game.map.tiles):
            for c, geom in enumerate(row):
                self.text_module.text_to_screen(self.screen, f'{(r, c)}', geom.x - 12, geom.y - 3,
                                                size=10, color=(0, 0, 0))

    def _draw_polygon_alpha(self, color, points, surface=None):
        lx, ly = zip(*points)
        min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
        target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.polygon(shape_surf, color, [(x - min_x, y - min_y) for x, y in points])

        self.screen.blit(shape_surf, target_rect)
        # surface.blit(shape_surf, target_rect)

    # @profile
    def _update_grid(self, ):
        for r, row in enumerate(self.game.map.tiles):
            for c, tile in enumerate(row):
                terrain_image = self.images.tiles[tile.terrain.image_path]
                terrain_image = pygame.transform.scale(terrain_image, DEFAULT_TERRAIN_IMAGE_SIZE)
                self.screen.blit(terrain_image,
                                 (tile.x - DEFAULT_TERRAIN_IMAGE_SIZE[0] / 2,
                                  tile.y - DEFAULT_TERRAIN_IMAGE_SIZE[1] / 2))

        # Coloring countries territory
        for player in self.game.players:
            for city in player.cities:
                for tile_coord in city.territory:
                    geom = self.game.map.get(*tile_coord).geometry

                    # borders
                    for n_index, neighbour in enumerate(self.game.map.get_neighbours_grid_coords(*tile_coord)):

                        if neighbour is None or self.game.map.whom_cell_is_it(self.game, *neighbour) != player.nation:
                            pygame.draw.line(self.screen, Nations.COLORS[player.nation],
                                             geom.points[n_index], geom.points[(n_index + 1) % 6], width=5)

        for r, row in enumerate(self.game.map.tiles):
            for c, tile in enumerate(row):
                pygame.draw.polygon(self.screen, BLACK, tile.points, 1)

    def _update_units_and_cities(self):
        for player in self.game.players:
            for city in player.cities:
                city.draw(self.screen, self.game, self.text_module)

            for unit in player.units:
                unit.draw(self.screen, self.game)

    def _update_paths(self):
        obj_selected = self.game.get_selected_unit() or self.game.get_selected_city()

        if obj_selected is None:
            return

        ranged_target = obj_selected.get_ranged_target(self.game)

        if ranged_target is None and len(obj_selected.path) == 0:
            return

        if ranged_target is None and isinstance(obj_selected, City):
            return

        if ranged_target is not None:
            unit_enemy = ranged_target

            from_hex = self.game.map.get(obj_selected.r, obj_selected.c).geometry
            target_hex = self.game.map.get(unit_enemy.r, unit_enemy.c).geometry
            self.draw_arrow(pygame.Vector2(from_hex.x, from_hex.y),
                            pygame.Vector2(target_hex.x, target_hex.y),
                            pygame.Color(0, 0, 0))
        else:
            # mp_used = 0
            mp_left = obj_selected.mp
            last_step_number = 0
            path_r_prev, path_c_prev = obj_selected.path[0]

            indices = []
            for i, (path_r, path_c) in enumerate(obj_selected.path[1:]):

                move_cost = self.game.map.get_data_edge((path_r_prev, path_c_prev), (path_r, path_c))['cost']

                if mp_left < move_cost:
                    step_number = last_step_number + 1
                    mp_left = obj_selected.mp_base
                    mp_left -= move_cost
                    indices.append(i-1)
                else:
                    step_number = last_step_number
                    mp_left -= move_cost

                last_step_number = step_number
                path_r_prev, path_c_prev = path_r, path_c

            for i, (path_r, path_c) in enumerate(obj_selected.path[1:]):
                path_hex = self.game.map.get(path_r, path_c).geometry

                if i in indices or (i == len(obj_selected.path) - 2):
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
        for ui_element in self.game.ui.ui_elements:
            ui_element.draw(self.screen, self.game, self.text_module)

        units = self.game.get_current_player().game_objects

        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (50, 600, 500, 175),
            width=0)

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (50, 600, 500, 175),
            width=2)

        unit = next((u for u in units if u.is_selected), None)
        if unit is not None:
            self.screen.blit(unit.image_ui, (55, 605))

            self.text_module.text_to_screen(self.screen,
                                            f'{unit.name}\nHP: {unit.hp}\nMP: {unit.mp}',
                                            x=55,
                                            y=605 + UI_UNIT_IMAGE_SIZE[1] + 1,
                                            size=20)

            enemy_unit = self.game.get_current_player().enemy_unit
            if enemy_unit is None:
                return

            self.screen.blit(enemy_unit.image_ui, (405, 605))
            self.text_module.text_to_screen(self.screen,
                                            f'{enemy_unit.name}\nHP: {enemy_unit.hp}\nMP: {enemy_unit.mp}',
                                            x=405,
                                            y=605 + UI_UNIT_IMAGE_SIZE[1] + 1,
                                            size=20)

        # self.game.next_turn_button.draw(self.screen, self.game, self.text_module)
        # self.game.show_moves_marker.draw(self.screen, self.game, self.text_module)

    def update_texts(self):
        if self.screen is None:
            return

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

    def draw_arrow(
            self,
            start: pygame.Vector2,
            end: pygame.Vector2,
            color: pygame.Color,
            body_width: int = 5,
            head_width: int = 15,
            head_height: int = 15,
    ):
        if self.screen is None:
            return

        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - head_height

        # Create the triangle head around the origin
        head_verts = [
            pygame.Vector2(0, head_height / 2),  # Center
            pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
            pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
        ]
        # Rotate and translate the head into place
        translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
        for i in range(len(head_verts)):
            head_verts[i].rotate_ip(-angle)
            head_verts[i] += translation
            head_verts[i] += start

        pygame.draw.line(self.screen, color, start, end, body_width)

        pygame.draw.line(self.screen, color, end, head_verts[1], body_width)
        pygame.draw.line(self.screen, color, end, head_verts[2], body_width)