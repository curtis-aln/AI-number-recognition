import pygame as pg
import numpy as np

from pygame import Vector2 as Vec2
from pygame import Rect

from settings import *

from text_renderer import TextRenderer
from interactive_box import InteractiveBox
from neural_network import NeuralNetworkRenderer, FeedForwardNeuralNetwork


class MouseHandler:
    def __init__(self) -> None:
        self.mouse_states = (False, False, False)


    def update_mouse_variables(self):
        self.mouse_states = pg.mouse.get_pressed()

    

class Simulation(MouseHandler):
    def __init__(self) -> None:
        super().__init__()

        self.window_dims = Vec2(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.window = pg.display.set_mode(self.window_dims, flags=pg.NOFRAME, vsync=False)
        self.clock  = pg.time.Clock()

        # runtime variables
        self.running = True
        self.paused = False

        # on-screen text management
        self.large_text = TextRenderer(self.window, 36)
        self.regular_text = TextRenderer(self.window)

        # other classes
        bounds = Rect(50, 180, 400, 400) # TODO: make bounds relative
        grid_resolution = Vec2(30, 30)
        self.interactive_box = InteractiveBox(self.window, bounds, grid_resolution)

        # neural networking
        inputs = int(grid_resolution[0] * grid_resolution[1])
        shape = np.array([inputs, 128, 64, 10])
        bounds = Rect(500, 180, 200, 400)
        self.neural_net = FeedForwardNeuralNetwork(shape)
        self.net_renderer = NeuralNetworkRenderer(self.window, self.neural_net, bounds)


    def handle_mouse_press(self):
        self.update_mouse_variables()

        if self.mouse_states[0]:
            self.interactive_box.handle_click()

        elif self.mouse_states[2]:
            self.interactive_box.handle_click(remove_mode=True)


    def run(self) -> None:
        while self.running:
            self.clock.tick(FRAME_RATE)
            self.handle_events()
            self.update_simulation()
            self.render_simulation()

    
    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False
                
                elif event.key == pg.K_SPACE:
                    self.paused = not self.paused
                

                elif event.key == pg.K_c:
                    self.interactive_box.clear_grid()

                elif event.key == pg.K_n:
                    self.interactive_box.add_noise(NOISE_STRENGTH)
                
                elif event.key == pg.K_b:
                    self.interactive_box.add_blur()

                elif event.key == pg.K_i:
                    self.interactive_box.inverse_grid()


    def update_simulation(self):
        self.handle_mouse_press()

        inputs = self.interactive_box.get_inputs()
        self.neural_net.feed_forward(inputs)


    """ rendering """
    def render_simulation(self):
        self.window.fill(WINDOW_COLOR)

        self.render_title()
        self.interactive_box.render()
        self.net_renderer.render_network()
        self.net_renderer._render_bounds()

        pg.display.update()
    

    def render_title(self):
        frame_rate = round(self.clock.get_fps(), 2)
        self.large_text.draw_text("Image Recognition AI", (40, 40), (255, 255, 255))
        self.regular_text.draw_text(f"frame rate: {frame_rate}", (40, 70), (255, 255, 255))
        
    


