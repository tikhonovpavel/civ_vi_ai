import pygame
import math
import networkx as nx



# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Initialize Pygame
pygame.init()

# Set the dimensions of the screen
screen_width = 800
screen_height = 600

screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill((255,255,255))

radius = 20

pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.

def text_to_screen(screen, text, x, y, size = 50,
            color = (200, 000, 000)):# 'data/fonts/orecrusherexpand.ttf'):
    try:
        text = str(text)        
        my_font = pygame.font.SysFont('Arial', size)
        text = my_font.render(text, True, color)
        screen.blit(text, (x, y))

    except Exception as e:
        print('Font Error, saw it coming')
        raise e

class Hexagon:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        
        self.points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = angle_deg * (math.pi / 180)
            point_x = x + radius * math.cos(angle_rad)
            point_y = y + radius * math.sin(angle_rad)
            self.points.append((point_x, point_y))

    def update(self):
        pygame.draw.polygon(screen, GREEN, self.points, 0)
        pygame.draw.polygon(screen, BLACK, self.points, 1)
        

class HexagonGrid:
    def __init__(self, rows, columns, offset_x=50, offset_y=50) -> None:

        self.n_rows = rows
        self.n_columns = columns

        self.hexagons = []
        self._graph = nx.DiGraph()

        side = math.sqrt(3) / 2 * radius
        
        for r in range(rows):
            row = []
            
            for c in range(columns):
                if r % 2 == 0:
                    row.append(Hexagon(3* radius * c + offset_x, side * r + offset_y))
                else:
                    row.append(Hexagon(1.5*radius * (c*2+1) + offset_x, side * r + offset_y))
                
                self._graph.add_node((r, c))
                
            self.hexagons.append(row)

        for r, row in enumerate(self.hexagons):
            for c, _ in enumerate(row):
                for neighbour in self.get_neighbours_grid_coords(r, c):
                    self._graph.add_edge((r, c), neighbour)
        

    def update(self, ):
        for r, row in enumerate(self.hexagons):
            for c, hex in enumerate(row):
                hex.update()
                text_to_screen(screen, f'{(r, c)}', hex.x-12, hex.y-3, size=10)


    def get(self, r, c):
        return self.hexagons[r][c]


    def get_neighbours_grid_coords(self, r, c):
        result = []
        print(f'for {(r, c)} neighbours are:')
        if r % 2 == 0:
            result = [(r-1,c-1), (r+1,c-1), (r-2,c), (r+2,c), (r-1,c), (r+1,c)]
        else:
            result = [(r-2,c), (r-1,c), (r+1,c), (r+2,c), (r-1,c+1), (r+1,c+1)]

        print(result)
        for i in range(len(result)):
            res_r, res_c = result[i]

            result[i] = res_r % self.n_rows, res_c % self.n_columns

        print(result)
        print()

        return result

    def get_hexagon_grid_coords(self, x, y):
        min_dist, min_r, min_c = math.inf, None, None

        for r, row in enumerate(self.hexagons):
            for c, hex in enumerate(row):
                dist = math.dist((hex.x, hex.y), (x, y))
                if dist < min_dist and dist < radius:
                    min_dist = dist
                    min_r, min_c = r, c
        
        return min_r, min_c 


# Update the screen
pygame.display.update()

# Run the game loop
running = True

clock = pygame.time.Clock()

hexagon_grid = HexagonGrid(30, 12)
hexagon_grid.update()


# Update the screen
pygame.display.update()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 1 == left button
                
                hexagon_grid.update()
                min_r, min_c = hexagon_grid.get_hexagon_grid_coords(*event.pos)

                if min_r is not None:
                    hex = hexagon_grid.get(min_r, min_c)

                    pygame.draw.circle(screen, (0,0,128), (hex.x, hex.y), radius=8)

                    # print([a for a, _ in hexagon_grid._graph.in_edges((min_r, min_c))])
                    path = nx.shortest_path(hexagon_grid._graph, (min_r, min_c), (15, 5))
                    print(path)
                    for path_r, path_c in path:
                        path_hex = hexagon_grid.get(path_r, path_c)
                        pygame.draw.circle(screen, (128,0,128), (path_hex.x, path_hex.y), radius=5)
                        

                    # for neighbour_r, neighbour_c in hexagon_grid.get_neighbours_grid_coords(min_r, min_c):
                    #     neighbour_hex = hexagon_grid.get(neighbour_r, neighbour_c)
                    #     pygame.draw.circle(screen, (128,0,128), (neighbour_hex.x, neighbour_hex.y), radius=5)
                            
                    pygame.display.update()
    # render(screen, hexagons)
    # clock.tick(50)

# Quit Pygame
pygame.quit()
