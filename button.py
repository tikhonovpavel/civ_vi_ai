import pygame


class ButtonStates:
    DEFAULT = 1
    HOVER = 2
    PRESSED = 3

class Button:
    def __init__(self, text, x, y, width, height, click_function,
                 color=(0, 128, 0), color_hover=(64, 64, 64), color_pressed=(128, 128, 128)):

        self.text = text

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.rect = pygame.Rect(x, y, width, height)

        self.color = color

        self.color_hover = color_hover
        self.color_pressed = color_pressed

        self.state = ButtonStates.DEFAULT

        self.click_function = click_function

    
    def draw(self, screen, game, text_module):
        pygame.draw.rect(
            screen,
            self.color,
            self.rect)
        text_module.text_to_screen(screen, self.text, x=self.x + 15, y=self.y + 15, size=30)
        # if self.state == ButtonStates.DEFAULT:
        #     button_color = self.color
        # elif self.state == ButtonStates.HOVER:
        #     button_color = self.color_hover
        # else:
        #     button_color = self.color_pressed
        #
        # pygame.draw.rect(
        #     screen,
        #     button_color,
        #     self.rect)
        # text_module.text_to_screen(screen, self.text, x=self.x+15, y=self.y+15, size=30)
