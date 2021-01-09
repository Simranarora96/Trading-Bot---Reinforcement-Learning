#!/usr/bin/env python
from env.trading_env import TradingEnv
import configparser
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import argparse
from datetime import datetime
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("configPath", help="Enter config path")
parser.add_argument("--seed", '-s', default="1", help='list of seeds to use separated by comma')
args = parser.parse_args()

path = args.configPath

env = DummyVecEnv([lambda: TradingEnv(path)])

config = configparser.ConfigParser()
config.read(path)

if config['MAIN']['Model'] == 'PPO2':
    model = PPO2(MlpPolicy, env, verbose=1, n_steps=int(config['MAIN']['NSteps']), tensorboard_log="./ppo_trading_tensorboard/")

model.learn(total_timesteps=int(config['MAIN']['TrainSteps']), tb_log_name=config['ENV']['ObsSpace']+config['ENV']['ActionSpace']+args.seed + str(datetime.now()))

model_save = "./Models/"+config['MAIN']['Model']+"-" + config['ENV']['TradingStrategy'] + config['ENV']['ObsSpace'] + config['ENV']['ActionSpace'] + str(datetime.now()) + ".h5"
model.save(model_save)
env.env_method("save_results")

