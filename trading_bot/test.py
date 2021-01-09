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

args = parser.parse_args()

path = args.configPath

env = DummyVecEnv([lambda: TradingEnv(path)])

config = configparser.ConfigParser()
config.read(path)

model = PPO2.load(config['MAIN']['Model'])

obs = env.reset()
for i in range(int(config['MAIN']['TestSteps'])):
    print(i)
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    
env.env_method("save_results")
