import argparse
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from agents import HumanAgent, ModelAgent, RandomAgent
from environment import QuantumTiqTaqToe
from sb3_contrib.common.wrappers import ActionMasker

from unitary.examples.tictactoe.enums import TicTacSquare

from utils import valid_fn




def play_agents(agent_X,
               agent_O,
               render_mode
               ):
    env = QuantumTiqTaqToe(render_mode = render_mode)
    env = ActionMasker(env, valid_fn)

    env.set_enemy_agent(agent_O)

    obs, _ = env.reset()
    while True:
        # Retrieve current action mask
        action_masks = valid_fn(env)
        action = agent_X.get_action(TicTacSquare.X,obs,action_masks)
        action = np.int32(action)
        obs, reward, terminated, _, _ = env.step(action)
        if(terminated):
            
            if(render_mode == "human"):
                print(reward)
            break
    
    return reward


def play(typeX, typeO, pathX = None, pathO = None):
    if(typeX == "human"):
        agent_X = HumanAgent()
    elif(typeX == "random"):
        agent_X = RandomAgent()
    else:
        agent_X = ModelAgent(pathX)

    if(typeO == "human"):
        agent_O = HumanAgent()
    elif(typeO == "random"):
        agent_O = RandomAgent()
    else:
        agent_O = ModelAgent(pathX)

    play_agents(agent_X, agent_O, render_mode="human")

     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum TIQ-TAQ-TOE self-play PPO')
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("-X", type=str, choices=["human","random","model"], default="random", help="type of agent for X")
    argParser.add_argument("-O", type=str, choices=["human","random","model"], default="random", help="type of agent for O")
    argParser.add_argument("-pathX",required = False, type=str, help="path for agent for X")
    argParser.add_argument("-pathO",required = False, type=str, help="path for agent for O")
    args = argParser.parse_args()
    play(args.X, args.O, args.pathX, args.pathO)
        