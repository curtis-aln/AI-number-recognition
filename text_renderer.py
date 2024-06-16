import pygame as pg

pg.font.init()

class TextRenderer:
    def __init__(self, screen, font_size=28):
        self.screen = screen
        self.font = pg.font.Font(None, font_size)  # You can choose any font and size here


    def draw_text(self, text, position, color=(255, 255, 255), centered=False):
        rendered_text = self.font.render(text, True, color)

        if centered:
            text_width, text_height = self.font.size(text)
            position = (position[0] - text_width // 2, position[1] - text_height // 2)

        self.screen.blit(rendered_text, position)