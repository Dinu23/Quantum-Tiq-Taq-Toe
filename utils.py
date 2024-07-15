import gymnasium as gym
import numpy as np
from unitary.examples.tictactoe.enums import TicTacSquare


enemy_map = {TicTacSquare.X: TicTacSquare.O, TicTacSquare.O: TicTacSquare.X}

pos_map = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8}



def valid_fn(env: gym.Env) -> np.ndarray:
    return env.get_valid()


def generate_map():
    action_to_direction = {}
    c = 'a'
    for i in range(9):
        action_to_direction[i] = chr(ord(c)+i) 
    val = 9
    for i in range(9):
        for j in range(9):
            if(i != j):
                action_to_direction[val] = chr(ord(c)+i)+ chr(ord(c)+j) 
                val+=1

    return action_to_direction


def generate_inverse_map():
    action_to_direction = {}
    c = 'a'
    for i in range(9):
        action_to_direction[chr(ord(c)+i)] = i 
    val = 9
    for i in range(9):
        for j in range(9):
            if(i != j):
                action_to_direction[chr(ord(c)+i)+ chr(ord(c)+j)] = val
                val+=1

    return action_to_direction
            