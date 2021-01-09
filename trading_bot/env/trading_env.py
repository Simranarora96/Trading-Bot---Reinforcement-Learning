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
        self.order_book = load_data(config['ENV']['Path'])
        self.max_no_of_securities = float(config['ENV']['MaxSecurities'])
        self.trading_strategy = config['ENV']['TradingStrategy']
        self.account_balance = float(config['ENV']['StartingMoney'])
        self.ask_price = 0
        self.bid_price = 0
        self.action = 0
        # Initialize Portfolio
        self.net_worth = self.account_balance
        self.no_of_securities_held = 0
        self.value_of_securities = 0
        self.money_borrowed = 0
        ask_price_list = self.order_book['L1ask_price'].values.tolist()
        bid_price_list = self.order_book['L1bid_price'].values.tolist()
        self.done = False
        self.action_space_map = {}
        self.obs_repr = config['ENV']['ObsSpace']
        self.action_repr = config['ENV']['ActionSpace']
        self.action_space = self.choose_actions()
        self.observation_space = self.choose_obs_space()

        self.state_vector = getattr(self, 'state_'+self.obs_repr)

    def choose_actions(self):
        if self.action_repr == "discrete":
            self.action_space_map = {0: -5, 1: -4, 2: -3, 3: -2, 4: -1, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5}
            return spaces.Discrete(11)

        elif self.action_repr == "continuous_1action":
            low = -np.ones(1)
            high = np.ones(1)
            self.action = [0]
            return spaces.Box(low, high, dtype=np.float32)

    @staticmethod
    def state_accb_nosec_bid_ask(self):
        return np.concatenate([(self.account_balance,), (self.no_of_securities_held,), (self.bid_price,), (self.ask_price,)])

    @staticmethod
    def state_accb_nosec_action_discrete_bid_ask(self):
        return np.concatenate([(self.account_balance,), (self.no_of_securities_held,) , (self.action,), (self.bid_price,), (self.ask_price,)])

    @staticmethod
    def state_accb_nosec_action_continous_bid_ask(self):
        return np.concatenate([(self.account_balance,), (self.no_of_securities_held,), (self.action), (self.bid_price,), (self.ask_price,)])

    def choose_obs_space(self):

        if self.obs_repr == "accb_nosec_bid_ask":
            obs_comp_sizes = [1, 1, 1, 1]
            obs_dim = np.sum(obs_comp_sizes)
            obs_high = np.ones(obs_dim)
            obs_low = -np.ones(obs_dim)

            # account balance
            obs_high[0:1] = float('inf')
            obs_low[0:1] = 0

            # No of securities
            obs_high[1:2] = self.max_no_of_securities
            obs_low[1:2] = 0

            # Bid Price
            obs_high[2:3] = float('inf')
            obs_low[2:3] = 0

            # Ask Price
            obs_high[3:4] = float('inf')
            obs_low[3:4] = 0

        elif self.obs_repr == "accb_nosec_action_discrete_bid_ask":
            obs_comp_sizes = [1, 1, 1, 1, 1]
            obs_dim = np.sum(obs_comp_sizes)
            obs_high = np.ones(obs_dim)
            obs_low = -np.ones(obs_dim)

            # account balance
            obs_high[0:1] = float('inf')
            obs_low[0:1] = 0

            # No of securities
            obs_high[1:2] = self.max_no_of_securities
            obs_low[1:2] = 0

            # Action Discrete
            obs_high[2:3] = 10
            obs_low[2:3] = 0

            # Bid Price
            obs_high[3:4] = float('inf')
            obs_low[3:4] = 0

            # Ask Price
            obs_high[4:5] = float('inf')
            obs_low[4:5] = 0

        elif self.obs_repr == "accb_nosec_action_continous_bid_ask":
            obs_comp_sizes = [1, 1, 1, 1, 1]
            obs_dim = np.sum(obs_comp_sizes)
            obs_high = np.ones(obs_dim)
            obs_low = -np.ones(obs_dim)

            # account balance
            obs_high[0:1] = float('inf')
            obs_low[0:1] = 0

            # No of securities
            obs_high[1:2] = self.max_no_of_securities
            obs_low[1:2] = 0

            # Action Continuous
            obs_high[2:3] = obs_high[2:3]
            obs_low[2:3] = obs_low[2:3]

            # Bid Price
            obs_high[3:4] = float('inf')
            obs_low[3:4] = 0

            # Ask Price
            obs_high[4:5] = float('inf')
            obs_low[4:5] = 0

        return spaces.Box(obs_low, obs_high, dtype=np.float32)

    def execute_trade(self, trading_strategy, trade_direction_and_quantity):
        # Calculate your current net_worth
        current_net_worth = self.account_balance + (self.no_of_securities_held * (self.bid_price + self.ask_price) / 2)
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
                if self.account_balance >= trade_direction_and_quantity * self.ask_price:
                    self.account_balance -= trade_direction_and_quantity * self.ask_price
                    self.no_of_securities_held += trade_direction_and_quantity

                elif current_net_worth >= trade_direction_and_quantity * self.ask_price:
                    self.money_borrowed += current_net_worth
                    self.account_balance += current_net_worth
                    self.account_balance -= trade_direction_and_quantity * self.ask_price
                    self.no_of_securities_held += trade_direction_and_quantity
                else:
                    action_penalty = trade_direction_and_quantity * self.ask_price - current_net_worth
                    self.done = True

            elif trade_direction_and_quantity < 0:
                # Sell action where action value is the quantity to be bought
                '''
                Sell action does the following things:
                1) Check if you can sell the securities asked by the agent to sell
                2) If not, sell whatever you can (this is aggressive sell.)
                '''
                if self.no_of_securities_held <= 0:
                    action_penalty = -trade_direction_and_quantity * self.bid_price
                elif self.no_of_securities_held < -trade_direction_and_quantity:
                    self.account_balance += self.no_of_securities_held * self.bid_price
                    self.no_of_securities_held = 0
                else:
                    self.account_balance += (-trade_direction_and_quantity) * self.bid_price
                    self.no_of_securities_held -= (-trade_direction_and_quantity)

        elif trading_strategy == "simple_buy_sell_at_midpoint":
            mid_point = (self.bid_price + self.ask_price) / 2
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
                    self.no_of_securities_held += trade_direction_and_quantity

                elif current_net_worth >= trade_direction_and_quantity * mid_point:
                    self.money_borrowed += current_net_worth
                    self.account_balance += current_net_worth
                    self.account_balance -= trade_direction_and_quantity * mid_point
                    self.no_of_securities_held += trade_direction_and_quantity
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
                if self.no_of_securities_held <= 0:
                    action_penalty = -trade_direction_and_quantity * mid_point
                elif self.no_of_securities_held < -trade_direction_and_quantity:
                    self.account_balance += self.no_of_securities_held * mid_point
                    self.no_of_securities_held = 0
                else:
                    self.account_balance += (-trade_direction_and_quantity) * mid_point
                    self.no_of_securities_held -= (-trade_direction_and_quantity)

        # CALCULATING NET WORTH AND VALUE OF SECURITIES HELD
        # (BID PRICE + ASK) /2
        self.value_of_securities = self.no_of_securities_held * (self.bid_price + self.ask_price) / 2
        self.net_worth = self.account_balance + self.value_of_securities - self.money_borrowed
        return action_penalty

    def step(self, action):
        self.action = action
        self.done = False
        if self.action_repr == "discrete":
            trade_direction_and_quantity = self.action_space_map[action]
        elif self.action_repr == "continuous_1action":
            trade_direction_and_quantity = int(10 * action)

        self.ask_price = ask_price_list.pop(0)
        self.bid_price = bid_price_list.pop(0)
        old_net_worth = copy.deepcopy(self.net_worth)

        action_penalty = self.execute_trade(self.trading_strategy, trade_direction_and_quantity)

        # CALCULATING REWARD
        reward = ((self.net_worth - old_net_worth - action_penalty) / old_net_worth) * 100

        # obs = np.array([self.account_balance, self.no_of_securities_held, bid_price_list[0], ask_price_list[0]])

        if config['ENV']['Debug']:
            global track_net_worth, track_reward, track_actions, total_reward, track_account_balance, track_held_sec, track_val_sec
            track_net_worth.append(self.net_worth)
            total_reward += reward
            track_reward.append(reward)
            track_actions.append(trade_direction_and_quantity)
            track_account_balance.append(self.account_balance)
            track_held_sec.append(self.no_of_securities_held)
            track_val_sec.append(self.no_of_securities_held * ((self.bid_price + self.ask_price) / 2))

        obs = self.state_vector(self)
        return obs / max(obs), reward, self.done, {}

    def reset(self):
        global ask_price_list, bid_price_list, config
        # Initialize ENV Params
        self.order_book = load_data(config['ENV']['Path'])
        self.max_no_of_securities = float(config['ENV']['MaxSecurities'])
        self.trading_strategy = config['ENV']['TradingStrategy']
        self.account_balance = float(config['ENV']['StartingMoney'])
        self.ask_price = 0
        self.bid_price = 0
        self.action = 0
        # Initialize Portfolio
        self.net_worth = self.account_balance
        self.no_of_securities_held = 0
        self.value_of_securities = 0
        self.money_borrowed = 0
        ask_price_list = self.order_book['L1ask_price'].values.tolist()
        bid_price_list = self.order_book['L1bid_price'].values.tolist()
        self.done = False
        self.action_space_map = {}
        self.obs_repr = config['ENV']['ObsSpace']
        self.action_repr = config['ENV']['ActionSpace']
        self.action_space = self.choose_actions()
        self.observation_space = self.choose_obs_space()

        self.state_vector = getattr(self, 'state_' + self.obs_repr)
        return self.state_vector(self)

    def render(self, mode='human'):
        print("Current Balance", self.account_balance)
        print("Net Worth", self.net_worth)
        print("Action", self.action)
        print("No of Securities", self.no_of_securities_held)

    def env_method(self, method_name):
        self.plot_results()

    def save_results(self):
        global track_net_worth, track_reward, track_actions, total_reward, track_account_balance, track_held_sec, track_val_sec
        print(total_reward)
        frames = [pd.DataFrame({'Net_Worth' : track_net_worth}), pd.DataFrame({'Reward' : track_reward}), pd.DataFrame({'Actions' : track_actions}),
                  pd.DataFrame({'Account_Balance' : track_account_balance}), pd.DataFrame({'Sec': track_held_sec}), pd.DataFrame({'Val_Sec':track_val_sec})]
        df = pd.concat(frames, axis=1)
        df.to_csv("Results/result" + config['ENV']['TradingStrategy'] + self.obs_repr + self.action_repr + str(datetime.now()) + ".csv")
