import numpy as np
from collections import defaultdict
from tqdm import tqdm
from texas_holdem import Action

class CFRTrainer:
    def __init__(self, game):
        """
        Initialize CFR trainer
        """
        self.game = game
        self.regrets = defaultdict(lambda: np.zeros(5)) # 5 actions
        self.strategy = defaultdict(lambda: np.ones(5) / 5) # Initial uniform strategy
        self.strategy_sum =defaultdict(lambda: np.zeros(5))
        self.iteractions = 0

    def get_strategy(self, info_set):
        """
        Get current strategy for an info set
        """
        regrets = self.regrets[info_set]
        positive_regrets = np.maximum(regrets, 0)

        # Normalize 
        regret_sum = np.sum(positive_regrets)
        if regret_sum > 0:
            return positive_regrets / regret_sum
        else:
            return np.ones(5) / 5 # uniformif no positive regrets
        
    def train(self, iterations=1000):
        """
        Run CFR algorithm for spectified iterations
        """
        for i in tqdm(range(iterations)):
            self._cfr_iteration()
            self.iteractions += 1
    
    def _cfr_iteration(self):
        """
        Run one iteration of CFR
        """
        # Reset the game
        self.game.reset()

        # Call recursive CFR
        self._cfr_recursive(1,1)
    
    def _cfr_recursive(self, p0, p1):
        """
        Recursive CFR implementation
        """
        # Check if game is over
        if self.game.phase == 'showdown':
            return self.game._calculate_rewards()
        
        #get current players
        current_player = self.game.current_player

        #create info set identifier
        if current_player == 0:
            info_set = self._get_info_set(0)
        else:
            info_set = self._get_info_set(1)
        
        #get valid actions
        valid_actions = self._get_valid_actions()

        #get current strategy
        strategy = self.get_strategy(info_set)

        # initialize action values
        action_values = np.zeros(5)

        # Try each action
        for action in valid_actions:
            action_idx = action.value

            #crate a copy of the game state
            game_copy = self._clone_game()

            # apply action
            amount = 0
            if action == Action.Raise:
                # Simple bet sizing for CFR
                player = game_copy.players[current_player]
                max_bet = player['stack'] + player['bet']
                current_max_bet = max(p['bet'] for p in game_copy.players)
                amount = current_max_bet * 2 # simple pot sized bet
                amount = min(amount, max_bet)
            
            # Take action in the game
            _, reward, done, _ = game_copy.step(action, amount)

            # Recursive call with updated probabilities
            if current_player == 0:
                action_values[action_idx] = self._cfr_recursive(p0 * strategy[action_idx], p1)
            else:
                action_values[action_idx] = self._cfr_recursive(p0, p1 * strategy[action_idx])
            
        # Calculate expected value
        ev = np.sum(strategy * action_values)

        # Calculate regrets
        if current_player == 0:
            regrets = action_values - ev
            self.regrets[info_set] += p1 * regrets
            self.strategy_sum[info_set] += p0 * strategy
        else:
            regrets = action_values - ev
            self.regrets[info_set] += p0 * regrets
            self.strategy_sum[info_set] += p1 * strategy

        return ev

    def _get_info_set(self, player_idx):
        """
        Create a string reprensents of an info set
        this is a simplified version - in practice include more game set up
        """
        player = self.game.players[player_idx]

        # Include players cards
        cards_str = ''.join(str(card) for card in player['hand'])

        # Include community cards
        community_str = ''.join(str(card) for card in self.game.community_cards)

        # Include phase
        phase_str = self.game.phase

        # Include pot and bets
        pot_str = str(self.game.pot)
        bet_str = str(player['bet'])

        # combine all info
        return f"{cards_str}|{community_str}|{phase_str}|{pot_str}|{bet_str}"

    def _get_valid_actions(self):
        from train import get_valid_actions
        return get_valid_actions(self.game, self.game.current_player)
    
    def _clone_game(self):
        import copy
        return copy.deepcopy(self.game)
    
    def get_average_strategy(self):
        avg_strategy = {}

        for info_set, strategy_sum in self.strategy_sum.items():
            total = np.sum(strategy_sum)
            if total > 0:
                avg_strategy[info_set] = strategy_sum / total
            else:
                avg_strategy[info_set] = np.ones(5) / 5
        
        return avg_strategy
