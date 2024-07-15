import argparse
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from typing import Callable
from agents import ModelAgent, ModelAgentV2
from custom_cnn import CustomCNN
from environment import QuantumTiqTaqToeV2
from sb3_contrib.common.wrappers import ActionMasker
import torch
from utils import enemy_map, valid_fn
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from unitary.examples.tictactoe.enums import (
    TicTacSquare,
    TicTacRules
)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def train_agents(
          rules = TicTacRules.QUANTUM_V3,
          measurment: bool = True,
          moves: bool = True,
          network: list = [32,32,16],
          N: int = 100,
          timestemps: int = 10000,
          n_steps: int = 1024,
          m: float = 1.0,
          lr: float = 0.0003,
          linear_lr: bool = True,
          folder: str = "models",
          logs: str = "logs",
          verbose: int = 1
):
    model_path = folder + "/model_rules_"+ str(rules.name) +"_measuremnt_"  + str(measurment) + "_moves_" + str(moves) + "_network_" + str(network)+ "_N_"  +str(N) + "_timestemps_" + str(timestemps)+"_modifier_" + str(m) +"_n_steps_" + str(n_steps) + "_lr_" + str(lr) + "_linerar_" + str(linear_lr) +"_itr_"
    env = QuantumTiqTaqToeV2(rules=rules,measurment=measurment, moves=moves)
    env = ActionMasker(env, valid_fn)
    if(linear_lr):
        lr = linear_schedule(lr)
    policy_kwargs = dict(net_arch=network)
    model = MaskablePPO('MultiInputPolicy',
                         env,
                         verbose = verbose,
                         policy_kwargs = policy_kwargs, 
                         n_steps=n_steps, 
                         learning_rate=lr, 
                         tensorboard_log=logs
                         )
    player = TicTacSquare.X
    
    eval_callback = MaskableEvalCallback(env, eval_freq=n_steps, n_eval_episodes=10)
    callbacks = [eval_callback]
    for i in range(N):
        if(verbose > 0):
             print(f"Train iteration:{i}, player: {player.name}")
        
        if(player == TicTacSquare.O):
            time = int(timestemps*m)
        else:
            time = timestemps
        
        model.learn(total_timesteps = time, tb_log_name=f"player_{player.name}", reset_num_timesteps=False,callback=callbacks,)
        saved_model_path = model_path + str(i+1)
        model.save(saved_model_path)
        
        player = enemy_map[player]
        enemy_agent = ModelAgentV2(saved_model_path)
        env.set_player(player)
        env.set_enemy_agent(enemy_agent)




if __name__ == '__main__':
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    parser = argparse.ArgumentParser(description='Quantum TIQ-TAQ-TOE self-play PPO')
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--rules", type=str, default="V1")
    argParser.add_argument("--measuremnt", type=bool, default=True, action=argparse.BooleanOptionalAction)
    argParser.add_argument("--moves", type=bool, default=True, action=argparse.BooleanOptionalAction)
    argParser.add_argument("--network", type=list_of_ints, default=[32,32,16])
    argParser.add_argument("-N","--N", type=int, default=100, help="number of enemy change")
    argParser.add_argument("-t","--timestemps", type=int, default=10000, help="timestemps to train the model during a single run")
    argParser.add_argument("-s","--no_steps", type=int, default=1024, help="no_steps")
    
    argParser.add_argument("-lr","--lr", type=float, default=0.01, help="learning rate")
    argParser.add_argument("--linear", type=bool, default=True, help="linear lr", action=argparse.BooleanOptionalAction)
    argParser.add_argument("-m","--modifier", type=float, default=1, help="how much to train more O player")
    
    argParser.add_argument("-f","--folder", type=str, default="models", help="folder to save models")
    argParser.add_argument("-l","--logs", type=str, default='logs', help="logs file")

    argParser.add_argument("-v","--verbose", type=int, default=1, help="verbose")
    argParser.add_argument("-d","--device", type=str, default='cpu', choices=['cpu', 'cuda'], help="device")
    args = argParser.parse_args()
    torch.set_default_device(args.device)   

    rules = None
    if(args.rules == "V1"):
        rules = TicTacRules.QUANTUM_V1
    if(args.rules == "V2"):
        rules = TicTacRules.QUANTUM_V2
    if(args.rules == "V3"):
        rules = TicTacRules.QUANTUM_V3
    print(args.measuremnt)
    print(args.moves)
    print(args.network)
    train_agents(
        rules= rules,
        measurment = args.measuremnt,
        moves = args.moves,
        network= args.network,
        N = args.N,
        timestemps = args.timestemps,   
        n_steps = args.no_steps,
        lr = args.lr,
        linear_lr= args.linear,
        m= args.modifier,
        folder = args.folder,
        logs = args.logs,
        verbose= args.verbose
    )
    
        