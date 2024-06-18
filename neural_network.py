# the nerual network will be a very basic feed forward neural network (FFNN).
# it will be trained using back propagation

import numpy as np
import pygame as pg

from numpy.random import uniform
from pygame import Rect
from pygame import Vector2 as Vec2
from interactive_box import value01_to_color, draw_rect_outline
from settings import *

INIT_WEIGHT_RANGE = 0.1
INIT_BIAS_RANGE = 0.1

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class FeedForwardNeuralNetwork:
    def __init__(self, shape : np.ndarray) -> None:
        self.shape = shape
        self._init_network()


    def _init_network(self) -> None:
        self.weights = []
        self.biases = []
        self.states = []

        self.states.append(np.zeros(self.shape[0]))

        for layer_idx in range(1, self.shape.size):
            weights = uniform(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE, (self.shape[layer_idx], self.shape[layer_idx-1]))
            biases  = uniform(-INIT_BIAS_RANGE, INIT_BIAS_RANGE, self.shape[layer_idx])

            self.weights.append(weights)
            self.biases.append(biases)
            self.states.append(biases * 0)
    

    # todo: store outputs of each layer
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        self.states[0] = inputs
        layer_index = 1
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            self.states[layer_index] = sigmoid(np.dot(layer_weights, self.states[layer_index-1]) + layer_biases)
            layer_index += 1

        return self.states[-1]



NODE_RADIUS = 2
WEIGHT_SHOW_PERCENT = 4 # x% of weights are shown
WEIGHT_WIDTH = 1


class NeuralNetworkRenderer:
    def __init__(self, render_surface : pg.Surface, network : FeedForwardNeuralNetwork, bounds : Rect) -> None:
        self.render_surface = render_surface
        self.network = network
        self.bounds = bounds

        # NOTE the first layer does not need to be rendered as that is shown by the interactive box
        self.inputless_shape = np.array([self.network.shape[i] for i in range(1, self.network.shape.size)])
        self._init_rendering()
        self._calculate_weight_graphics()
    

    def _init_rendering(self):
        self.positions = []
        inputless_shape = self.inputless_shape

        x_spacing = self.bounds.width / (inputless_shape.size - 1)
        for layer_idx in range(1, inputless_shape.size):
            layer = []
            # calculating spacing for the nodes in this layer
            y_spacing = self.bounds.height / (inputless_shape[layer_idx] - 1)
            for node_idx in range(inputless_shape[layer_idx]):
                position_center = Vec2(self.bounds.x + x_spacing * (layer_idx-1), 
                                       self.bounds.y + y_spacing * node_idx)
                layer.append(position_center)
            self.positions.append(layer)
    


    def _calculate_weight_graphics(self):
        shape = self.network.shape
        inputless_shape = self.inputless_shape
        
        self.weight_surface = pg.surface.Surface(self.bounds.size)
        self.weight_surface.fill(WINDOW_COLOR)


        #for layer_idx in range(1, inputless_shape.size):
        #    for node1_idx in range(inputless_shape[layer_idx - 1]): # -1 as we are lookiking at the previous nodes
        #        for node2_idx in range(inputless_shape[layer_idx]): # the current notes
        #            if np.random.uniform(0, 1) < WEIGHT_SHOW_PERCENT: # only rendering some of the weights
        #                color = value_to_color(self.network.weights[layer_idx][node2_idx][node1_idx])
#
        #                pos1 = self.positions[layer_idx - 2][node1_idx] - Vec2(self.bounds.topleft)
        #                pos2 = self.positions[layer_idx - 1][node2_idx] - Vec2(self.bounds.topleft)
        #                pg.draw.line(self.weight_surface, color, pos1, pos2, WEIGHT_WIDTH)


    def render_network(self):
        # rendering all of the weights
        self.render_surface.blit(self.weight_surface, (self.bounds.x, self.bounds.y))

        # rendering all of the nodes
        for layer_idx in range(1, self.inputless_shape.size):
            for node_idx in range(self.inputless_shape[layer_idx]):
                color = value01_to_color(self.network.states[layer_idx][node_idx])
                pg.draw.circle(self.render_surface, color, self.positions[layer_idx - 1][node_idx], NODE_RADIUS)
        
    

    # used for debugging
    def _render_bounds(self):
        draw_rect_outline(self.render_surface, self.bounds)