import argparse
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from agents import HumanAgent, ModelAgent, ModelAgentV2, RandomAgent
from environment import QuantumTiqTaqToe, QuantumTiqTaqToeV2
from sb3_contrib.common.wrappers import ActionMasker

from unitary.examples.tictactoe.enums import TicTacSquare
from unitary.examples.tictactoe.enums import TicTacRules

from utils import valid_fn,enemy_map




def play_agents(
               rules,
               agent_X,
               agent_O,
               render_mode,
               measurmentX = True,
               movesX = True,
               measurmentO = True,
               movesO = True
               ):
    env = QuantumTiqTaqToeV2(rules=rules,render_mode = render_mode)
    env = ActionMasker(env, valid_fn)

    env.set_enemy_agent(agent_O)

    obs, _ = env.reset()
    count = 0
    player = TicTacSquare.X
    while True:
        # Retrieve current action mask
        action_masks = valid_fn(env)
        if(player == TicTacSquare.X):
            observation = dict()
            if(measurmentX):
                observation["measurement"] = obs["measurement"]
            if(movesX):
                observation["moves"] = obs["moves"]
            
            action = agent_X.get_action(TicTacSquare.X,observation,action_masks)
            action = np.int32(action)
        else:
            observation = dict()
            if(measurmentO):
                observation["measurement"] = obs["measurement"]
            if(movesO):
                observation["moves"] = obs["moves"]
            
            action = agent_O.get_action(TicTacSquare.O,observation,action_masks)
            action = np.int32(action)

        count +=1
        obs, reward, terminated, _, _ = env.play_step(player,action)
        if(terminated):
            if(render_mode == "human"):
                print(reward,count)
            break
        player = enemy_map[player]
    return reward,count


def play(
        version,
        typeX,
        typeO,
        pathX = None,
        pathO = None,
        measurmentX = True,
        movesX = True,
        measurmentO = True,
        movesO = True):
    if(typeX == "human"):
        agent_X = HumanAgent()
    elif(typeX == "random"):
        agent_X = RandomAgent()
    else:
        agent_X = ModelAgentV2(pathX)




    if(typeO == "human"):
        agent_O = HumanAgent()
    elif(typeO == "random"):
        agent_O = RandomAgent()
    else:
        agent_O = ModelAgentV2(pathO)

    if(version == 1):
        rules = TicTacRules.QUANTUM_V1
    if(version == 3):
        rules = TicTacRules.QUANTUM_V3
    play_agents(rules,agent_X, agent_O, render_mode="human",measurmentX=measurmentX,measurmentO=measurmentO,movesX=movesX,movesO=movesO)

     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum TIQ-TAQ-TOE self-play PPO')
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-V", type=int, choices=[1,3], default=3)
    argParser.add_argument("-X", type=str, choices=["human","random","model"], default="random", help="type of agent for X")
    argParser.add_argument("-O", type=str, choices=["human","random","model"], default="random", help="type of agent for O")
    argParser.add_argument("-pathX",required = False, type=str, help="path for agent for X")
    argParser.add_argument("-pathO",required = False, type=str, help="path for agent for O")
    argParser.add_argument("--measurmentX",default=True, type=bool,  action=argparse.BooleanOptionalAction)
    argParser.add_argument("--movesX",default=True, type=bool,  action=argparse.BooleanOptionalAction)
    argParser.add_argument("--measurmentO",default=True, type=bool,  action=argparse.BooleanOptionalAction)
    argParser.add_argument("--movesO",default=True, type=bool,  action=argparse.BooleanOptionalAction)
    
    args = argParser.parse_args()
    play(args.V, args.X, args.O, args.pathX, args.pathO,measurmentX=args.measurmentX, movesX=args.movesX, measurmentO=args.measurmentO, movesO=args.movesO,)
        