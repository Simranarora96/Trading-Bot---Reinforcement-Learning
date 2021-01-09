import gym
from gym import spaces
import pandas as pd
import numpy as np
import configparser
import copy
from datetime import datetime

ask_price_list = []
bid_price_list = []

config = configparser.ConfigParser()


# DEBUG
track_net_worth = []
total_reward = 0
track_reward = []
track_actions = []
track_held_sec = []
track_account_balance = []
track_val_sec = []


def load_data(path):
    return pd.read_csv(path)


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path):
        global ask_price_list, bid_price_list, config
        # Initialize ENV Params
        config.read(path)
        self.max_no_of_securities = float(config['ENV']['MaxSecurities'])
        self.trading_strategy = config['ENV']['TradingStrategy']
        self.account_balance = float(config['ENV']['StartingMoney'])
        self.commodities = str(config['ENV']['Commodities']).split(",")
        self.ask_price = [0]*len(self.commodities)
        self.bid_price = [0]*len(self.commodities)
        # Initialize Portfolio
        self.net_worth = self.account_balance
        self.no_of_securities_held = [0]*len(self.commodities)
        self.value_of_securities = [0]*len(self.commodities)
        self.money_borrowed = 0
        for commodity in self.commodities:
            data = pd.read_csv(config['ENV']['Path']+commodity+'.csv')
            ask_price_list.append(data['L1ask_price'].values.tolist())
            bid_price_list.append(data['L1bid_price'].values.tolist())
        self.done = False
        self.action_space_map = {}
        self.obs_repr = config['ENV']['ObsSpace']
        self.action_repr = config['ENV']['ActionSpace']
        self.action_space = self.choose_actions()
        self.observation_space = self.choose_obs_space()
        self.state_vector = getattr(self, 'state_'+self.obs_repr)

    def choose_actions(self):
        if self.action_repr == "continuous_1action":
            return spaces.Box(low=np.array([-1] * len(self.commodities)), high=np.array([1] * len(self.commodities)), dtype=np.float32)

    @staticmethod
    def state_accb_nosec_bid_ask(self):
        return np.concatenate([(self.account_balance,), (self.no_of_securities_held), (self.bid_price), (self.ask_price)])

    def choose_obs_space(self):

        if self.obs_repr == "accb_nosec_bid_ask":
            obs_comp_sizes = [1, len(self.commodities), len(self.commodities), len(self.commodities)]
            obs_dim = np.sum(obs_comp_sizes)
            obs_high = np.ones(obs_dim)
            obs_low = -np.ones(obs_dim)
            # account balance
            obs_high[0:1] = float('inf')
            obs_low[0:1] = 0

            # No of securities
            obs_high[1:len(self.commodities)+1] = [self.max_no_of_securities]*len(self.commodities)
            obs_low[1:len(self.commodities)+1] = [0]*len(self.commodities)
            
            # Bid Price
            obs_high[len(self.commodities)+1:2*len(self.commodities)+1] = float('inf')
            obs_low[len(self.commodities)+1:2*len(self.commodities)+1] = 0

            # Ask Price
            obs_high[2*len(self.commodities)+1:3*len(self.commodities)+1] = float('inf')
            obs_low[2*len(self.commodities)+1:3*len(self.commodities)+1] = 0

        return spaces.Box(obs_low, obs_high, dtype=np.float32)

    def execute_trade(self, trading_strategy, trade_direction_and_quantity, index):
        # Calculate your current net_worth
        current_net_worth = self.account_balance + (self.no_of_securities_held[index] * (self.bid_price[index] + self.ask_price[index]) / 2)
        action_penalty = 0
        # This is a terminal state check, ending episode with penalty if your net worth <=0
        if current_net_worth <= 0:
            self.done = True

        # Run different trading strategy
        elif trading_strategy == "simple_buy_sell":
            if trade_direction_and_quantity > 0:
                # Buy action where action value is the quantity to be bought
                '''
                Buying action does the following things
                1) Check if you have the money to execute the trade
                2) If you don not have sufficient funds check if you can borrow money equal to your net worth that would
                 be enough to do the trade
                3) Else penalize as you cannot perform the action.
                '''
                if self.account_balance >= trade_direction_and_quantity * self.ask_price[index]:
                    self.account_balance -= trade_direction_and_quantity * self.ask_price[index]
                    self.no_of_securities_held[index] += trade_direction_and_quantity

                elif current_net_worth >= trade_direction_and_quantity * self.ask_price[index]:
                    self.money_borrowed += current_net_worth
                    self.account_balance += current_net_worth
                    self.account_balance -= trade_direction_and_quantity * self.ask_price[index]
                    self.no_of_securities_held[index] += trade_direction_and_quantity
                else:
                    action_penalty = trade_direction_and_quantity * self.ask_price[index] - current_net_worth
                    self.done = True

            elif trade_direction_and_quantity < 0:
                # Sell action where action value is the quantity to be bought
                '''
                Sell action does the following things:
                1) Check if you can sell the securities asked by the agent to sell
                2) If not, sell whatever you can (this is aggressive sell.)
                '''
                if self.no_of_securities_held[index] <= 0:
                    action_penalty = -trade_direction_and_quantity * self.bid_price[index]
                elif self.no_of_securities_held[index] < -trade_direction_and_quantity:
                    self.account_balance += self.no_of_securities_held[index] * self.bid_price[index]
                    self.no_of_securities_held[index] = 0
                else:
                    self.account_balance += (-trade_direction_and_quantity) * self.bid_price[index]
                    self.no_of_securities_held[index] -= (-trade_direction_and_quantity)

        elif trading_strategy == "simple_buy_sell_at_midpoint":
            mid_point = (self.bid_price[index] + self.ask_price[index]) / 2
            if trade_direction_and_quantity > 0:
                # Buy action where action value is the quantity to be bought
                '''
                Buying action does the following things
                1) Check if you have the money to execute the trade
                2) If you don not have sufficient funds check if you can borrow money equal to your net worth that would
                 be enough to do the trade
                3) Else penalize as you cannot perform the action.
                '''
                if self.account_balance >= trade_direction_and_quantity * mid_point:
                    self.account_balance -= trade_direction_and_quantity * mid_point
                    self.no_of_securities_held[index] += trade_direction_and_quantity

                elif current_net_worth >= trade_direction_and_quantity * mid_point:
                    self.money_borrowed += current_net_worth
                    self.account_balance += current_net_worth
                    self.account_balance -= trade_direction_and_quantity * mid_point
                    self.no_of_securities_held[index] += trade_direction_and_quantity
                else:
                    action_penalty = trade_direction_and_quantity * mid_point - current_net_worth
                    self.done = True

            elif trade_direction_and_quantity < 0:
                # Sell action where action value is the quantity to be bought
                '''
                Sell action does the following things:
                1) Check if you can sell the securities asked by the agent to sell
                2) If not, sell whatever you can (this is aggressive sell.)
                '''
                if self.no_of_securities_held[index] <= 0:
                    action_penalty = -trade_direction_and_quantity * mid_point
                elif self.no_of_securities_held[index] < -trade_direction_and_quantity:
                    self.account_balance += self.no_of_securities_held[index] * mid_point
                    self.no_of_securities_held[index] = 0
                else:
                    self.account_balance += (-trade_direction_and_quantity) * mid_point
                    self.no_of_securities_held[index] -= (-trade_direction_and_quantity)

        # CALCULATING NET WORTH AND VALUE OF SECURITIES HELD
        # (BID PRICE + ASK) /2
        self.value_of_securities[index] = self.no_of_securities_held[index] * (self.bid_price[index] + self.ask_price[index]) / 2
        self.net_worth = self.account_balance + self.value_of_securities[index] - self.money_borrowed
        return action_penalty

    def step(self, action):
        print(action)
        self.action_space = action
        self.done = False
        action_penalty = 0
        val_sec = 0
        old_net_worth = copy.deepcopy(self.net_worth)
        for i in range(len(self.commodities)):
            if self.action_repr == "continuous_1action":
                trade_direction_and_quantity = int(10 * action[i])
            self.ask_price[i] = ask_price_list[i].pop(0)
            self.bid_price[i] = bid_price_list[i].pop(0)
            action_penalty += self.execute_trade(self.trading_strategy, trade_direction_and_quantity, i)
            val_sec += self.no_of_securities_held[i] * ((self.bid_price[i] + self.ask_price[i]) / 2)
            if self.done:
                break

        # CALCULATING REWARD
        reward = ((self.net_worth - old_net_worth - action_penalty) / old_net_worth) * 100

        if config['ENV']['Debug']:
            global track_net_worth, track_reward, track_actions, total_reward, track_account_balance, track_held_sec, track_val_sec
            track_net_worth.append(self.net_worth)
            total_reward += reward
            track_reward.append(reward)
            track_account_balance.append(self.account_balance)
            track_held_sec.append(sum(self.no_of_securities_held))
            track_val_sec.append(val_sec)

        obs = self.state_vector(self)
        return obs / max(obs), reward, self.done, {}

    def reset(self):
        global ask_price_list, bid_price_list, config
        # Initialize ENV Params
        self.max_no_of_securities = float(config['ENV']['MaxSecurities'])
        self.trading_strategy = config['ENV']['TradingStrategy']
        self.account_balance = float(config['ENV']['StartingMoney'])
        self.commodities = str(config['ENV']['Commodities']).split(",")
        self.ask_price = [0]*len(self.commodities)
        self.bid_price = [0]*len(self.commodities)
        # Initialize Portfolio
        self.net_worth = self.account_balance
        self.no_of_securities_held = [0]*len(self.commodities)
        self.value_of_securities = [0]*len(self.commodities)
        self.money_borrowed = 0
        for commodity in self.commodities:
            data = pd.read_csv(config['ENV']['Path']+commodity+'.csv')
            ask_price_list.append(data['L1ask_price'].values.tolist())
            bid_price_list.append(data['L1bid_price'].values.tolist())
        self.done = False
        self.action_space_map = {}
        self.obs_repr = config['ENV']['ObsSpace']
        self.action_repr = config['ENV']['ActionSpace']
        self.action_space = self.choose_actions()
        self.observation_space = self.choose_obs_space()
        self.state_vector = getattr(self, 'state_'+self.obs_repr)
        return self.state_vector(self)

    def render(self, mode='human'):
        print("Current Balance", self.account_balance)
        print("Net Worth", self.net_worth)
        print("Action", self.action_space)
        print("No of Securities", self.no_of_securities_held)

    def env_method(self, method_name):
        self.plot_results()

    def save_results(self):
        global track_net_worth, track_reward, track_actions, total_reward, track_account_balance, track_held_sec, track_val_sec
        print(total_reward)
        frames = [pd.DataFrame({'Net_Worth' : track_net_worth}), pd.DataFrame({'Reward' : track_reward}),
                  pd.DataFrame({'Account_Balance' : track_account_balance}), pd.DataFrame({'Sec': track_held_sec}), pd.DataFrame({'Val_Sec':track_val_sec})]
        df = pd.concat(frames, axis=1)
        df.to_csv("Results/result" + config['ENV']['TradingStrategy'] + self.obs_repr + self.action_repr + str(datetime.now()) + ".csv")
