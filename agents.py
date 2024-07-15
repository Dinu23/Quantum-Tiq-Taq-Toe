
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from unitary.examples.tictactoe.enums import TicTacSquare

from utils import generate_inverse_map


class Agent():
    def __init__(self):
        pass

    def get_action(self, player, observation, valid_action_mask):
        pass


class RandomAgent(Agent):
    def __init__(self):
        pass

    def get_action(self, player, observation, valid_action_mask):
        valid_action_mask = np.array(valid_action_mask)
        valid_action_mask = valid_action_mask / np.sum(valid_action_mask)
        action = np.random.choice(len(valid_action_mask),1,p=valid_action_mask)[0]
        return action
        
class ModelAgent(Agent):
    def __init__(self, model_file):
        self._model_file = model_file
        self._model = MaskablePPO.load(self._model_file)

    def get_action(self, player, observation, valid_action_mask):
        if(player == TicTacSquare.O):
                observation = np.array([observation[0],observation[2],observation[1]])


        valid_action_mask = np.array(valid_action_mask)
        action, _states  = self._model.predict(observation, action_masks=valid_action_mask)
        action = np.int32(action)
        return action
    
        
class ModelAgentV2(Agent):
    def __init__(self, model_file):
        self._model_file = model_file
        self._model = MaskablePPO.load(self._model_file)

    def get_action(self, player, observation, valid_action_mask):
        if(player == TicTacSquare.O):
            if("measurement" in  observation.keys()):
                observation["measurement"] = np.array([observation["measurement"][0],observation["measurement"][2],observation["measurement"][1]])
            if("moves" in  observation.keys()):
                observation["moves"] = np.array([observation["moves"][1],observation["moves"][0]])



        valid_action_mask = np.array(valid_action_mask)
        action, _states  = self._model.predict(observation, action_masks=valid_action_mask)
        action = np.int32(action)
        return action
    

class HumanAgent(Agent):
    def __init__(self):
        self._inverse_map = generate_inverse_map()
        pass

    def get_action(self, player, observation, valid_action_mask):
        while(True):
            move = input(f'Player {player} to move: ')
            
            action = self._inverse_map[move]

            if(valid_action_mask[action] == 1):
                return action
            
            print("Invalid action")