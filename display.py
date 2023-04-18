import pygame

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

HEX_DEFAULT = (0, 255, 0)
HEX_SELECTED = (0, 128, 0)

# Set the size for the image
DEFAULT_UNIT_IMAGE_SIZE = (30, 30)
UI_UNIT_IMAGE_SIZE = (100, 100)

DEFAULT_NATION_IMAGE_SIZE = (15, 10)
UI_NATION_IMAGE_SIZE = (50, 50)


pygame.font.init()
def text_to_screen(screen, text, x, y, size=50,
            color = (200, 000, 000)):
    try:
        text = str(text)        
        my_font = pygame.font.SysFont('Arial', size)
        text = my_font.render(text, True, color)
        screen.blit(text, (x, y))

    except Exception as e:
        print('Font Error, saw it coming')
        raise e


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


    def update_all(self):#players, hexagon_grid, paths):
        self._update_grid(self.game.hexagon_grid)
        self._update_units(self.game.players, self.game.hexagon_grid)
        self._update_paths(self.game.paths, self.game.hexagon_grid)
        self._update_ui(self.game.players[0].units)

        pygame.display.update()


    def _update_grid(self, hexagon_grid):
        
        for r, row in enumerate(hexagon_grid.hexagons):
            for c, hex in enumerate(row):
                pygame.draw.polygon(self.screen, HEX_DEFAULT, hex.points, 0)
                pygame.draw.polygon(self.screen, BLACK, hex.points, 1)

                text_to_screen(self.screen, f'{(r, c)}', hex.x-12, hex.y-3, size=10)
        
        # pygame.display.update()


    def _update_units(self, players, hexagon_grid):

        for player in players:
            # player.flag

            for unit in player.units:
                unit_hex = hexagon_grid.get(unit.r, unit.c)

                if unit.selected:
                    pygame.draw.polygon(self.screen, HEX_SELECTED, unit_hex.points, 0)
                    pygame.draw.polygon(self.screen, BLACK, unit_hex.points, 1)

                nation_image = nation_icons[player.nation]['default']
                self.screen.blit(unit.image, (unit_hex.x - unit.image.get_width() / 2, unit_hex.y - unit.image.get_height() / 2))
                self.screen.blit(nation_image, (unit_hex.x - nation_image.get_width() / 2, unit_hex.y + 7))

                
                
                hp_offset = 15
                hp_length = 16
                hp_thickness = 5
                
                # pygame.draw.rect(self.screen, (240,240,240), (unit_hex.x - hp_length / 2, unit_hex.y - hp_offset, hp_length, 0))
                pygame.draw.rect(
                    self.screen,
                    (255,0,0),
                    (unit_hex.x - hp_length / 2, unit_hex.y - hp_offset, hp_length * (unit.hp / 100), hp_thickness),
                    )
                pygame.draw.rect(
                    self.screen,
                    (0,0,0),
                    (unit_hex.x - hp_length / 2, unit_hex.y - hp_offset, hp_length, hp_thickness),
                    width=1)
                


            

        # pygame.display.update()

    def _update_paths(self, paths, hexagon_grid):

        for path in paths:
            for i, (path_r, path_c) in enumerate(path):
                radius = 8 if i == 0 else 5

                path_hex = hexagon_grid.get(path_r, path_c)
                pygame.draw.circle(self.screen, (128,0,128), (path_hex.x, path_hex.y), radius=radius)

    def _update_ui(self, units):
        pygame.draw.rect(
            self.screen,
            (255,255,255),
            (50, 600, 500, 150),
            width=0)


        pygame.draw.rect(
            self.screen,
            (0,0,0),
            (50, 600, 500, 150),
            width=2)
        
        unit = next((u for u in units if u.selected), None)
        if unit is not None:
            self.screen.blit(unit.image_ui, (55, 605))
            text_to_screen(self.screen,
                            f'HP: {unit.hp}',
                              x=55,#+UI_UNIT_IMAGE_SIZE[0]+5, 
                              y=605+UI_UNIT_IMAGE_SIZE[1]+5,
                              size=30)
            


            enemy_unit = self.game.get_current_player().ready_to_attack_unit
            if enemy_unit is None:
                return
                        
            self.screen.blit(enemy_unit.image_ui, (405, 605))
            text_to_screen(self.screen,
                f'HP: {enemy_unit.hp}',
                    x=405,#+UI_UNIT_IMAGE_SIZE[0]+5, 
                    y=605+UI_UNIT_IMAGE_SIZE[1]+5,
                    size=30)
                
            
            