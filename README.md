# Trading-Bot---Reinforcement-Learning
Designed an Open AI GYM based trading environment in Python that enabled training on multiple commodities, different trading strategies, and a reward system based on PnL

# Offline Learning

In this project we try to simulate the real world trading environment to create our own trading bots. Our trading bot would try to learn the hidden trends and with enough training would learn a trading strategy that would generate profit over it’s net worth. We simulate the market maker through an Open AI Gym environment and execute various trading strategies like buy at ask and sell at bid along with trading at mid point price.

The framework presented in this project is decoupled into various components that allow for training utilizing:
-- Different trading strategies
-- Various combination of Observation and Action Space
-- PnL Reward System
-- Multiple commodities
-- Different Algorithms
-- Multiple seeds

## File Structure
```
env
  |- trading_env_multi.py
  |- trading_env.py
Data
  |- Commodities
    | - All training commodities (.csv)
Configs
  |- Contains config files for diffferent experiments (.ini)
Launcher
  |- Contains shell scripts to run multiple seeds for training (.sh)
Models
  |- Contains the trained model (.h5)
Notebooks
  |- Contains notebooks to visulaize results generated in logging
Results
  |- Contains all logged results from training (.csv)
ppo_trading_tensorboard
  |- Logs tensorboard data 
test.py - Used to test the model saved after training
train.py - Trains the model on a single commodity and uses the single commodity env (trading_env.py). Allows 2 actions space and 3 observation space
train_multi.py - Trains the model on single and multiple commodities and uses trading_env_multi.py. Allows 1 action space and 1 observation space.
```

**Note - Naming convention of files - Files in Results, Configs, Launcher and Models folder follow a naming convention of modelname_(other details if necessary)_tradingstrategy_actionspace_observationspace_(timestamp if needed).(ini, csv, h5, sh)**
## Explanation of Config and Launcher file
```
Config.ini

[ENV]
Path = */Path to the training data*/
TradingStrategy = */Mention trading strategy [simple_buy_sell, simple_buy_sell_at_midpoint] */
MaxSecurities = */Currently not used in env*/
StartingMoney = */Money for the bot to start trading with*/
Debug = */Logs various portfolio parameters during training if set to 1*/
ObsSpace = */Mention type of obs space to be used [eg. accb_nosec_bid_ask] */
ActionSpace = */Mention type of action space to be used [eg. continuous_1action] */
Commodities = */Mention name of different commodities [eg. ZNH0:MBO,ZTH0:MBO,ZFH0:MBO] */
[MAIN]
Model = */Mention model name [eg. PPO2] */
TrainSteps = */Training Steps = 1447200(Should be divisible by NSteps) */
NSteps = */ N Steps = 3600*/
```
```
Launcher.sh
#!/bin/bash
parallel  ./train_multi.py ./Configs/ppo_multi10_trading_simple_continuous_action.ini \
--seed {1} \
::: {1..4} */ Specify the no of seeds to run, here it is 4*/
```

## Training a Bot
```
Steps
1) Create a Config and a Launcher file or use an existing Launcher, Config file.
2) Navigate to the folder(gym_trading) containing all the above mentioned folders & run the following commands:

$ chmod +x Launcher/<name of launcher file.sh> (Only needed for new launcher files)
$ ./Launcher/<name of launcher file.sh> (This runs multiple seeds and traning starts)

--- New Terminal[To start tensor board] ---
Navigate to the project directory and run:
$ tensorboard --logdir ./ppo_trading_tensorboard/
```
