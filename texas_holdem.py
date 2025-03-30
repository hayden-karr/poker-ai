# texas_holdem.py
import numpy as np
from poker_env import Deck, Card, Suit
from enum import Enum

class Action(Enum):
    Fold = 0
    Check = 1
    Call = 2
    Raise = 3
    All_in = 4

class PokerGame:
    def __init__(self, num_players=2, small_blind=1, big_blind=2, initial_stack=100):
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = initial_stack
        self.deck = Deck
        self.reset()

    def reset(self):
        """
        Reset game to the initial state
        """
        self.deck.reset()
        self.pot = 0
        self.community_cards = []
        self.current_players = 0
        self.dealer_position = 0
        self.phase = 'pre_flop' # pre_flop, flop, turn, river, showdown 

        # Initialize players with chips and empty hands
        self.players = []
        for i in range(self.num_players):
            self.players.append({
                'stack': self.initial_stack,
                'hand': [],
                'bet': 0,
                'folder': False,
                'all_in': False,
            })
        
        # Deal hole cards
        for _ in range(2):
            for player in self.players:
                player['hand'].append(self.deck.deal())

        # Post Blinds
        sb_pos = (self.dealer_position + 1)
        bb_pos = (self.dealer_position + 2)

        self.players[sb_pos]['stack'] -= self.small_blind
        self.players[sb_pos]['bet'] = self.small_blind
        self.players[bb_pos]['stack'] -= self.big_blind
        self.players[bb_pos]['bet'] = self.big_blind

        self.pot = self.small_blind + self.big_blind
        self.current_players = (bb_pos + 1) % self.num_players
        self.betting_round_active = True

        return self._get_observation(self.current_players)
    
    def step(self, action, amount=0):
        """
        Execute an action in the game
        action: One of fold, check, call, raise, all-in
        amount: Bet amount ( only on raise)
        
        Returns: (next_observation, reward, done, info)
        """

        player = self.players[self.current_players]
        current_max_bet = max(p['bet'] for p in self.players)
        current_call_amount = current_max_bet - player['bet']

        # Process an action
        if action == Action.Fold:
            player['folded'] =True
        
        elif action == Action.Check:
            # Check is valid if no one has bet yet
            if current_call_amount > 0:
                raise ValueError("Cannot check as there is an existing bet")

        elif action == Action.Call:
            # Call the current bet 
            player['stack'] -= current_call_amount
            player['bet'] += current_call_amount
            self.pot += current_call_amount
        
        elif action == Action.Raise:
            # Validate and raise
            if amount <= current_max_bet:
                raise ValueError("Raise amount must be larger than the current bet")
            if amount > player['stack'] + player['bet']:
                raise ValueError("Cannot raise more than in your stack")

            # Calculate how much more to add
            additional_amount = amount - player['bet']
            player['stack'] -= additional_amount
            player['bet'] = amount
            self.pot += additional_amount
        
        elif action == Action.All_in:
            # Go all in with remaining chips in stack
            player['all_in'] = True
            self.pot += player['stack']
            player['bet'] +=player['stack']
            player['stack'] = 0
        
        # Move to the next player
        self._next_player()

        # Check if the betting round is over 
        if self._is_betting_round_over():
            self._advance_phase()
        
        # Get observations for next player
        observation = self._get_observation(self.current_player)
        done = self.phase == 'showdown'

        reward = 0
        if done:
            rewards = self._calculate_rewards()
            reward = rewards[self.current_player]
        
        info = {'phase': self.phase, 'pot': self.pot}

        return observation, reward, done, info
    
    def _next_player(self):
        """
        Move to the next active player 
        """
        self.current_player = (self.current_player +1) % self.num_players
        while (self.players[self.current_player]['folded'] or self.players[self.current_player]['all_in']):
            self.current_player = (self.current_player +1) % self.num_players
    
    def _is_betting_round_over(self):
        """
        Check if the current betting round is over
        """
        active_players = [p for p in self.players if not p['folded']]

        # If only one player left, betting is over
        if len(active_players) == 1:
            return True

        # Check if all active players have the same bet
        bet_amounts = [p['bet'] for p in active_players if not p['all_in']]
        return len(set(bet_amounts)) <= 1

    def _advance_phase(self):
        """
        Move to the next phase of the game
        """
        if self.phase == 'pre_flop':
            # Deal the flop
            self.community_cards.extend(self.deck.deal(3))
            self.phase = 'flop'
        elif self.phase == 'flop':
            # Deal the turn
            self.community_cards.append(self.deck.deal())
            self.phase = 'turn'
        elif self.phase == 'turn':
            # Deal the river
            self.community_cards.append(self.deck.deal())
            self.phase = 'river'
        elif self.phase == 'river':
            # Go to showdown
            self.phase = 'showdown'

        # Reset bets for this round
        for player in self.players:
            player['bet'] = 0

        # Reset to the first active player
        self.current_player = (self.dealer_position + 1) % self.num_players
        while (self.players[self.current_player]['folded'] or self.players[self.current_player]['all_in']):
            self.current_player = (self.current_player +1) % self.num_players
    
    def _get_observation(self, player_idx):
        """
        Create an observation for the specified player
        This is what the AI will see as the input
        """
        player = self.players[player_idx]

        # Create Observation dictionary
        obs = {
            'hand': player['hand'], # Only this players cards are visible
            'community_cards': self.community_cards,
            'pot': self.pot,
            'stack': player['stack'],
            'current_bet': player['bet'],
            'phase': self.phase,
            'position': player_idx,
            'dealer_position': self.dealer_position,
            'num_players': self.num_players,

            # info for the other players but only partially observable
            'players_info': []
        }

        # Add information about other players (without revealing their cards!!)
        for i, p in enumerate(self.players):
            if i != player_idx:
                obs['players_info'].append({
                    'position': i,
                    'stack': p['stack'],
                    'bet': p['bet'],
                    'folded': p['folded'],
                    'all_in': p['all_in']
                })
        
        return obs

    def _calculate_rewards(self):
        """
        Calculate rewards at showdown
        """
        # Initialize rewards array
        rewards = [0] * self.num_players

        # If all but one player folded, that player wins
        active_players = [i for i, p in enumerate(self.players) if not p['folded']]
        if len(active_players) == 1:
            winner = active_players[0]
            rewards[winner] = self.pot
            return rewards
        
        # Otherwise, evaluate hands to see who wins -  need to implement
        elif:
            # iterate through players hands and check to see if they have a good hand
            rewards[winner] = self.pot
            return rewards