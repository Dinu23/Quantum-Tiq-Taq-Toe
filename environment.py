import gymnasium as gym
import numpy as np
import time
from gymnasium import spaces
import stable_baselines3
from unitary.examples.tictactoe.enums import (
    TicTacSquare,
    TicTacResult,
    TicTacRules
)
from unitary.examples.tictactoe.tic_tac_toe import TicTacToe
from unitary.examples.tictactoe.tic_tac_toe import _histogram
import logging
from gymnasium import logger
logger.set_level(logging.ERROR)
from agents import Agent, RandomAgent
from unitary.examples.tictactoe.tic_tac_toe import _MARK_SYMBOLS
from utils import generate_map, enemy_map, pos_map

TIME = 2

class QuantumTiqTaqToe(gym.Env):

    def __init__(self, 
                 player = TicTacSquare.X, 
                 enemy_agent : Agent = RandomAgent(), 
                 rules: TicTacRules = TicTacRules.QUANTUM_V3, 
                 run_on_hardware: bool = False, 
                 render_mode=None
                 ):
        
        self.game = TicTacToe(rules, run_on_hardware)
        self.size = None
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3, 3), dtype=np.float32)
        self.player = player
        self.enemy_agent = enemy_agent
        self.render_mode = render_mode

        if(rules == TicTacRules.CLASSICAL):
            self.action_space = spaces.Discrete(9)
        else:
            self.action_space = spaces.Discrete(9+36*2)

        self._action_to_direction = generate_map()
        

    def set_player(self, player):
        self.player = player

    def set_enemy_agent(self, enemy_agent):
        self.enemy_agent = enemy_agent

    def _generate_gamestate(self, count = 100):
        results = self.game.board.peek(count=count)
        hist = _histogram(
            [
                [TicTacSquare.from_result(square) for square in result]
                for result in results
            ]
        )
        state = np.zeros((3,3,3))
        k=0
        for el in hist:
            state[0][k//3][k%3] = el[TicTacSquare.EMPTY]
            state[1][k//3][k%3] = el[TicTacSquare.X]
            state[2][k//3][k%3] = el[TicTacSquare.O]
            k+=1
        return state/count
    

    def get_valid(self):
        valid = []
        c = 'a'
        for i in range(9):
            if(chr(ord(c)+i) not in self.game.empty_squares):
                valid.append(0)
            else:
                valid.append(1)
        if(self.game.rules == TicTacRules.CLASSICAL):
            return valid
        for i in range(9):
            for j in range(9):
                if(i != j):
                    if ( ((chr(ord(c)+i) not in self.game.empty_squares) or (chr(ord(c)+j) not in self.game.empty_squares)) and (self.game.rules == TicTacRules.QUANTUM_V1)):
                        valid.append(0)
                    else:
                        valid.append(1)       
        return valid


    def _print_observation(self, observation):
        output = "\n"
        for row in range(3):
            for mark in TicTacSquare:
                output += " "
                for col in range(3):
                    output += f" {_MARK_SYMBOLS[mark]} {observation[mark.value][row][col]:.2f}"
                    if col != 2:
                        output += " |"
                output += "\n"
            
            output += "--------------------------\n"
        print(output)
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.clear()

        observation = self._generate_gamestate()
        if(self.render_mode =="human"):
            self._print_observation(observation)
            time.sleep(TIME)
            
        
        if(self.player == TicTacSquare.O):
            action = self._action_to_direction[self.enemy_agent.get_action(TicTacSquare.X, observation, self.get_valid())]
            self.game.move(action, TicTacSquare.X)
            observation = self._generate_gamestate()
            if(self.render_mode =="human"):
                print(f"{TicTacSquare.X.name}: {action}")
                self._print_observation(observation)
                time.sleep(TIME)
            
        return observation, {}

    


    def _compute_reward(self,outcome):
        if(outcome == TicTacResult.DRAW or outcome == TicTacResult.BOTH_WIN):
            return 0 
        elif(outcome == TicTacResult.X_WINS):
            if(self.player == TicTacSquare.X):
                return 1
            else:
                return -1 
        elif(outcome == TicTacResult.O_WINS):
            if(self.player == TicTacSquare.X):
                return -1
            else:
                return 1
        

    def step(self, action):
        action = self._action_to_direction[action]

        outcome = self.game.move(action, self.player)

        observation = self._generate_gamestate()
        if(self.render_mode =="human"):
            print(f"{self.player.name}: {action}")
            self._print_observation(observation)
            time.sleep(TIME)
        
        if(outcome != TicTacResult.UNFINISHED):
            terminated = 1
            reward =  self._compute_reward(outcome)    
        else:
            enemy = enemy_map[self.player]
            action = self._action_to_direction[self.enemy_agent.get_action(enemy, observation, self.get_valid())]
            outcome = self.game.move(action, enemy)

            observation = self._generate_gamestate()
            
            if(self.render_mode =="human"):
                print(f"{enemy.name}: {action}")
                self._print_observation(observation)
                time.sleep(TIME)


            if(outcome != TicTacResult.UNFINISHED):
                terminated = 1
                reward =  self._compute_reward(outcome)    
            else:
                terminated = 0
                reward = 0
        
            
        return observation, reward, terminated, False, {}
    

class QuantumTiqTaqToeV2(gym.Env):

    def __init__(self, 
                 player = TicTacSquare.X, 
                 enemy_agent : Agent = RandomAgent(), 
                 rules: TicTacRules = TicTacRules.QUANTUM_V3, 
                 run_on_hardware: bool = False, 
                 measurment: bool = True,
                 moves: bool = True,
                 render_mode = None
                 ):
        
        self.game = TicTacToe(rules, run_on_hardware)
        self.size = None

        self.measurment = measurment
        self.moves = moves
        observation_types = dict()
        if(measurment): 
            observation_types[ "measurement"] = spaces.Box(low=0, high=1, shape=(3, 3, 3), dtype=np.float32)
        if(moves):
            observation_types[ "moves"] = spaces.Box(low=0, high=10, shape=(2, 9, 9), dtype=np.int32)
        self.observation_space = spaces.Dict(observation_types)
           
        self.player = player
        self.enemy_agent = enemy_agent
        self.render_mode = render_mode
        if(rules == TicTacRules.CLASSICAL):
            self.action_space = spaces.Discrete(9)
        else:
            self.action_space = spaces.Discrete(9+36*2)

        self._last_observation = np.zeros((2,9,9))
        self._action_to_direction = generate_map()
        

    def set_player(self, player):
        self.player = player

    def set_enemy_agent(self, enemy_agent):
        self.enemy_agent = enemy_agent

    def _generate_gamestate(self, count = 100):
        results = self.game.board.peek(count=count)
        hist = _histogram(
            [
                [TicTacSquare.from_result(square) for square in result]
                for result in results
            ]
        )
        state = np.zeros((3,3,3))
        k=0
        for el in hist:
            state[0][k//3][k%3] = el[TicTacSquare.EMPTY]
            state[1][k//3][k%3] = el[TicTacSquare.X]
            state[2][k//3][k%3] = el[TicTacSquare.O]
            k+=1
        return state/count
    

    def get_valid(self):
        valid = []
        c = 'a'
        for i in range(9):
            if(chr(ord(c)+i) not in self.game.empty_squares):
                valid.append(0)
            else:
                valid.append(1)
        if(self.game.rules == TicTacRules.CLASSICAL):
            return valid
        for i in range(9):
            for j in range(9):
                if(i != j):
                    if ( ((chr(ord(c)+i) not in self.game.empty_squares) or (chr(ord(c)+j) not in self.game.empty_squares)) and (self.game.rules == TicTacRules.QUANTUM_V1)):
                        valid.append(0)
                    else:
                        valid.append(1)       
        return valid


    def _print_observation(self, observation):
        output = "\n"
        for row in range(3):
            for mark in TicTacSquare:
                output += " "
                for col in range(3):
                    output += f" {_MARK_SYMBOLS[mark]} {observation[mark.value][row][col]:.2f}"
                    if col != 2:
                        output += " |"
                output += "\n"
            
            output += "--------------------------\n"
        print(output)
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.clear()
        self._last_observation = np.zeros((2,9,9))
        table = self._generate_gamestate()
        
        if(self.render_mode =="human"):
            self._print_observation(table)
            print(self._last_observation)
            time.sleep(TIME)
    

        observation = dict()
        if(self.measurment):
            observation["measurement"] = table
        if(self.moves):
            observation["moves"] = self._last_observation

        
        if(self.player == TicTacSquare.O):
            action = self._action_to_direction[self.enemy_agent.get_action(TicTacSquare.X, observation, self.get_valid())]
            self.game.move(action, TicTacSquare.X)
            if(len(action) == 1):
                i = pos_map[action[0]]
                self._last_observation[0][i][i] +=1
            else:
                i = pos_map[action[0]]
                j = pos_map[action[1]]
                self._last_observation[0][i][j] +=1
            table = self._generate_gamestate()
            
            if(self.render_mode =="human"):
                self._print_observation(table)
                print(self._last_observation)
                time.sleep(TIME)

            observation = dict()
            if(self.measurment):
                observation["measurement"] = table
            if(self.moves):
                observation["moves"] = self._last_observation
            
        return observation, {}

    


    def _compute_reward(self,outcome):
        if(outcome == TicTacResult.DRAW or outcome == TicTacResult.BOTH_WIN):
            return 0 
        elif(outcome == TicTacResult.X_WINS):
            if(self.player == TicTacSquare.X):
                return 1
            else:
                return -1 
        elif(outcome == TicTacResult.O_WINS):
            if(self.player == TicTacSquare.X):
                return -1
            else:
                return 1
    
    def _check_matrix(self,matrix):
        l = []
        for i in range(3):
            for j in range(3):
                if((matrix[0][i][j] > 0 and matrix[0][i][j] < 1) or (matrix[1][i][j] > 0 and matrix[1][i][j] < 1) or (matrix[2][i][j] > 0 and matrix[2][i][j] < 1)):
                    return False, None
                if(matrix[1][i][j] == 1):
                    l.append((0,i*3+j))
                if(matrix[2][i][j] == 1):
                    l.append((1,i*3+j))
        
        return True,l

    def step(self, action):
        action = self._action_to_direction[action]
        outcome = self.game.move(action, self.player)
        table = self._generate_gamestate()

        done_measuremnt, locations = self._check_matrix(table)
        if(done_measuremnt):
            self._last_observation = np.zeros((2,9,9))
            for el in locations:
                self._last_observation[el[0]][el[1]][el[1]] +=1
        else:
            if(len(action) == 1):
                i = pos_map[action[0]]
                self._last_observation[self.player.value-1][i][i]+=1
            else:
                i = pos_map[action[0]]
                j = pos_map[action[1]]
                self._last_observation[self.player.value-1][i][j]+=1
            
       
        if(self.render_mode =="human"):
            print(f"{self.player.name}: {action}")
            print(self._last_observation)
            self._print_observation(table)
            time.sleep(TIME)
    

        observation = dict()
        if(self.measurment):
            observation["measurement"] = table
        if(self.moves):
            observation["moves"] = self._last_observation
        
        if(outcome != TicTacResult.UNFINISHED):
            terminated = 1
            reward =  self._compute_reward(outcome)    
        else:
            enemy = enemy_map[self.player]
            action = self._action_to_direction[self.enemy_agent.get_action(enemy, observation, self.get_valid())]
            outcome = self.game.move(action, enemy)

            table = self._generate_gamestate()

            done_measuremnt, locations = self._check_matrix(table)
            if(done_measuremnt):
                self._last_observation = np.zeros((2,9,9))
                for el in locations:
                    self._last_observation[el[0]][el[1]][el[1]] +=1
            else:
                if(len(action) == 1):
                    i = pos_map[action[0]]
                    self._last_observation[enemy.value-1][i][i]+=1
                else:
                    i = pos_map[action[0]]
                    j = pos_map[action[1]]
                    self._last_observation[enemy.value-1][i][j]+=1
                
            
            if(self.render_mode =="human"):
                print(f"{self.player.name}: {action}")
                print(self._last_observation)
                self._print_observation(table)
                time.sleep(TIME)


            observation = dict()
            if(self.measurment):
                observation["measurement"] = table
            if(self.moves):
                observation["moves"] = self._last_observation

            if(outcome != TicTacResult.UNFINISHED):
                terminated = 1
                reward =  self._compute_reward(outcome)    
            else:
                terminated = 0
                reward = 0
        
        # if(terminated == 1):
        #     print(self.player,reward)
            
        return observation, reward, terminated, False, {}



    def play_step(self,player,action):
        action = self._action_to_direction[action]
        outcome = self.game.move(action, player)
        table = self._generate_gamestate()

        done_measuremnt, locations = self._check_matrix(table)
        if(done_measuremnt):
            self._last_observation = np.zeros((2,9,9))
            for el in locations:
                self._last_observation[el[0]][el[1]][el[1]] +=1
        else:
            if(len(action) == 1):
                i = pos_map[action[0]]
                self._last_observation[self.player.value-1][i][i]+=1
            else:
                i = pos_map[action[0]]
                j = pos_map[action[1]]
                self._last_observation[self.player.value-1][i][j]+=1
    
        if(self.render_mode =="human"):
            print(f"{player.name}: {action}")
            print(self._last_observation)
            self._print_observation(table)
            time.sleep(TIME)

        reward = 0
        if(outcome == TicTacResult.X_WINS):
            reward =  1
        elif(outcome == TicTacResult.O_WINS):
            reward = -1

        terminated = 0
        if(outcome != TicTacResult.UNFINISHED):
            terminated = 1


        observation = dict()
        observation["measurement"] = table
        observation["moves"] = self._last_observation

        return observation, reward, terminated, False, {}