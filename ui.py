from typing import List

import pygame


class ButtonStates:
    DEFAULT = 1
    HOVER = 2
    PRESSED = 3


class UIElement:
    def __init__(self, text, x, y, rect, click_function):
        self.text = text
        self.x = x
        self.y = y
        self.rect = rect
        self.click_function = click_function

        self._state = None

    def draw(self, screen, game, text_module):
        pass

    def click(self):
        pass

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        raise NotImplementedError()


class Button(UIElement):
    def __init__(self, text, x, y, width, height, click_function, color=(0, 128, 0),
                 color_hover=(64, 64, 64), color_pressed=(128, 128, 128)):

        super().__init__(text, x, y, pygame.Rect(x, y, width, height), click_function)

        self.width = width
        self.height = height

        self.color = color

        self.color_hover = color_hover
        self.color_pressed = color_pressed

        self._state = ButtonStates.DEFAULT

    def draw(self, screen, game, text_module):
        pygame.draw.rect(
            screen,
            self._get_current_color(),
            self.rect)
        text_module.text_to_screen(screen, self.text, x=self.x + 15, y=self.y + 15, size=30)

    def click(self):
        self.click_function()

    def _get_current_color(self):
        if self._state == ButtonStates.DEFAULT:
            return self.color
        elif self._state == ButtonStates.HOVER:
            return self.color_hover
        elif self._state == ButtonStates.PRESSED:
            return self.color_pressed

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state in [ButtonStates.DEFAULT, ButtonStates.HOVER, ButtonStates.PRESSED]:
            self._state = state
        else:
            raise ValueError("Invalid state")


class Marker(UIElement):
    def __init__(self, text, x, y, click_function=None, on_enable_function=None, on_disable_function=None, state=False):
        super(Marker, self).__init__(text, x, y, pygame.Rect(x, y, 25, 25), click_function)
        self._state = state

        self.on_enable_function = on_enable_function
        self.on_disable_function = on_disable_function

    def draw(self, screen, game, text_module):
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        pygame.draw.rect(screen, (255, 255, 255),
                         (self.rect.x + 2, self.rect.y + 2, self.rect.width - 4, self.rect.height - 4))

        # Draw a check mark inside the square
        if self._state:
            pygame.draw.line(screen, (128, 128, 128), (self.x + 3, self.y + 12), (self.x + 10, self.y + 22), 5)
            pygame.draw.line(screen, (128, 128, 128), (self.x + 10, self.y + 22), (self.x + 35 - 10 - 3, self.y + 2), 5)

        text_module.text_to_screen(screen, self.text, x=self.x + 25 + 10, y=self.y - 6, size=30, color=(0, 0, 0))

    def click(self):
        self._state = not self._state

        if self._state and self.on_enable_function:
            self.on_enable_function()
        elif not self._state and self.on_disable_function:
            self.on_disable_function()

        self.click_function()


    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if isinstance(state, bool):
            self._state = state
        else:
            raise TypeError("Invalid state type")


class Text(UIElement):
    def __init__(self, text, x, y, width, height, click_function=None, color=(0, 0, 0)):

        super().__init__(text, x, y, pygame.Rect(x, y, width, height), click_function)

        self.text_template = text
        self.color = color
        self.click_function = click_function

    def draw(self, screen, game, text_module):
        text_module.text_to_screen(screen, self.text, x=self.x + 15, y=self.y + 15, size=30)

    def click(self):
        self.click_function()

    def update(self, **kwargs):
        self.text = self.text_template.format(**kwargs)


class UI:
    def __init__(self, ui_elements: List[UIElement]):
        self.ui_elements = ui_elements

    def append(self, ui_element: UIElement):
        self.ui_elements.append(ui_element)

    def screen_click(self, pos, display):
        for ui_element in self.ui_elements:
            if ui_element.rect.collidepoint(pos):
                ui_element.click()
                ui_element.draw(display.screen, display.game, display.text_module)


