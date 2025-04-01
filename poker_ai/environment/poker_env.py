# poker_env.py
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np


class Suit(Enum):
    """Card suits enumeration"""
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Card:
    """Represents a playing card with rank and suit"""
    
    def __init__(self, rank: int, suit: Suit):
        """
        Initialize a card
        
        Args:
            rank: Card rank (2-14, where 11=J, 12=Q, 13=K, 14=A)
            suit: Card suit from the Suit enum
        """
        if not 2 <= rank <= 14:
            raise ValueError(f"Card rank must be between 2 and 14, got {rank}")
        
        self.rank = rank
        self.suit = suit
    
    def __str__(self) -> str:
        """String representation of the card"""
        rank_str = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(self.rank, str(self.rank))
        suit_str = {
            Suit.HEARTS: '❤️', 
            Suit.DIAMONDS: '♦', 
            Suit.CLUBS: '♣', 
            Suit.SPADES: '♠'
        }[self.suit]
        return f"{rank_str}{suit_str}"
    
    def __repr__(self) -> str:
        """Developer representation of the card"""
        return f"Card({self.rank}, {self.suit.name})"
    
    def to_one_hot(self) -> np.ndarray:
        """Convert card to one-hot encoding (52-dimensional vector)"""
        one_hot = np.zeros(52)
        idx = self.suit.value * 13 + (self.rank - 2)
        one_hot[idx] = 1
        return one_hot


class Deck:
    """Represents a deck of playing cards"""
    
    def __init__(self):
        """Initialize a standard 52-card deck"""
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self) -> None:
        """Reset to a fresh, ordered deck of 52 cards"""
        self.cards = []
        for suit in Suit:
            for rank in range(2, 15):  # 2 through Ace (14)
                self.cards.append(Card(rank, suit))
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the deck"""
        np.random.shuffle(self.cards)

    def deal(self, n: int = 1) -> Union[Card, List[Card]]:
        """
        Deal n cards from the deck
        
        Args:
            n: Number of cards to deal
            
        Returns:
            A single Card if n=1, otherwise a list of Cards
            
        Raises:
            ValueError: If not enough cards left in the deck
        """
        if n > len(self.cards):
            raise ValueError(f"Not enough cards left in the deck. Requested {n}, have {len(self.cards)}")
        
        if n == 1:
            card = self.cards[0]
            self.cards = self.cards[1:]
            return card
        else:
            dealt_cards = self.cards[:n]
            self.cards = self.cards[n:]
            return dealt_cards
    
    def __len__(self) -> int:
        """Return the number of cards left in the deck"""
        return len(self.cards)


class Action(Enum):
    """Possible poker actions enumeration"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    ALL_IN = 4


class GamePhase(Enum):
    """Phases of a poker hand"""
    PRE_FLOP = "pre_flop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class Player:
    """Represents a poker player"""
    
    def __init__(self, stack: int = 100):
        """
        Initialize a player
        
        Args:
            stack: Starting chip count
        """
        self.stack: int = stack
        self.hand: List[Card] = []
        self.bet: int = 0
        self.folded: bool = False
        self.all_in: bool = False
        
    def reset(self, stack: Optional[int] = None) -> None:
        """
        Reset player for a new hand
        
        Args:
            stack: Optional new stack amount, otherwise keep current stack
        """
        if stack is not None:
            self.stack = stack
        self.hand = []
        self.bet = 0
        self.folded = False
        self.all_in = False
        
    def receive_card(self, card: Card) -> None:
        """
        Add a card to the player's hand
        
        Args:
            card: The card to add
        """
        self.hand.append(card)
        
    def place_bet(self, amount: int) -> int:
        """
        Place a bet of the specified amount
        
        Args:
            amount: Amount to bet
            
        Returns:
            The actual amount bet (may be less if player doesn't have enough chips)
            
        Raises:
            ValueError: If bet amount is negative or zero
        """
        if amount <= 0:
            raise ValueError(f"Bet amount must be positive, got {amount}")
            
        # Cap bet at available stack
        actual_amount = min(amount, self.stack)
        self.stack -= actual_amount
        self.bet += actual_amount
        
        # Check if player is all-in
        if self.stack == 0:
            self.all_in = True
            
        return actual_amount
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert player to dictionary representation (for observation)
        
        Returns:
            Dictionary with player information
        """
        return {
            'stack': self.stack,
            'bet': self.bet,
            'folded': self.folded,
            'all_in': self.all_in
        }


class PokerGame:
    """Texas Hold'em poker game implementation"""
    
    def __init__(self, num_players: int = 2, small_blind: int = 1, 
                 big_blind: int = 2, initial_stack: int = 100):
        """
        Initialize a poker game
        
        Args:
            num_players: Number of players
            small_blind: Small blind amount
            big_blind: Big blind amount
            initial_stack: Starting stack for each player
        """
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = initial_stack
        self.deck = Deck()
        self.players: List[Player] = []
        self.pot: int = 0
        self.community_cards: List[Card] = []
        self.current_player: int = 0
        self.dealer_position: int = 0
        self.phase: GamePhase = GamePhase.PRE_FLOP
        
        # Initialize players
        for _ in range(num_players):
            self.players.append(Player(initial_stack))
            
    def reset(self) -> Dict[str, Any]:
        """
        Reset the game to initial state
        
        Returns:
            Observation for the first player to act
        """
        # Reset deck and game state
        self.deck.reset()
        self.pot = 0
        self.community_cards = []
        self.phase = GamePhase.PRE_FLOP
        
        # Reset players
        for player in self.players:
            player.reset(self.initial_stack)
        
        # Deal hole cards
        for _ in range(2):
            for player in self.players:
                card = self.deck.deal()
                player.receive_card(card)
        
        # Post blinds
        sb_pos = (self.dealer_position + 1) % self.num_players
        bb_pos = (self.dealer_position + 2) % self.num_players
        
        # Small blind
        sb_amount = self.players[sb_pos].place_bet(self.small_blind)
        self.pot += sb_amount
        
        # Big blind
        bb_amount = self.players[bb_pos].place_bet(self.big_blind)
        self.pot += bb_amount
        
        # Set first player to act (after big blind)
        self.current_player = (bb_pos + 1) % self.num_players
        
        # Return observation
        return self._get_observation(self.current_player)
    
    def step(self, action: Action, amount: int = 0) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action in the game
        
        Args:
            action: Player action (fold, check, call, raise, all-in)
            amount: Bet amount (only used for raise)
            
        Returns:
            Tuple of (observation, reward, done, info)
            
        Raises:
            ValueError: If the action is invalid in the current state
        """
        player = self.players[self.current_player]
        current_max_bet = max(p.bet for p in self.players)
        call_amount = current_max_bet - player.bet
        
        # Process action
        if action == Action.FOLD:
            player.folded = True
            
        elif action == Action.CHECK:
            if call_amount > 0:
                raise ValueError("Cannot check when there is an existing bet")
                
        elif action == Action.CALL:
            actual_call = player.place_bet(call_amount)
            self.pot += actual_call
            
        elif action == Action.RAISE:
            if amount <= current_max_bet:
                raise ValueError(f"Raise amount must be greater than current bet of {current_max_bet}")
                
            # Calculate additional amount needed to raise
            raise_amount = amount - player.bet
            actual_raise = player.place_bet(raise_amount)
            self.pot += actual_raise
            
        elif action == Action.ALL_IN:
            all_in_amount = player.stack
            player.place_bet(all_in_amount)  # This will set all_in flag
            self.pot += all_in_amount
        
        # Move to next player
        self._next_player()
        
        # Check if betting round is over
        if self._is_betting_round_over():
            self._advance_phase()
        
        # Get observation for next player
        observation = self._get_observation(self.current_player)
        done = self.phase == GamePhase.SHOWDOWN
        
        # Calculate rewards if game is done
        reward = 0
        if done:
            rewards = self._calculate_rewards()
            reward = rewards[self.current_player]
        
        info = {'phase': self.phase.value, 'pot': self.pot}
        
        return observation, reward, done, info
    
    def _next_player(self) -> None:
        """Move to next active player"""
        self.current_player = (self.current_player + 1) % self.num_players
        while (self.players[self.current_player].folded or 
               self.players[self.current_player].all_in):
            self.current_player = (self.current_player + 1) % self.num_players
    
    def _is_betting_round_over(self) -> bool:
        """Check if current betting round is over"""
        active_players = [p for p in self.players if not p.folded]
        
        # If only one player left, betting is over
        if len(active_players) == 1:
            return True
            
        # Check if all active players have the same bet or are all-in
        non_all_in = [p for p in active_players if not p.all_in]
        if not non_all_in:
            return True
            
        bet_amounts = set(p.bet for p in non_all_in)
        return len(bet_amounts) <= 1
    
    def _advance_phase(self) -> None:
        """Move to next phase of the game"""
        if self.phase == GamePhase.PRE_FLOP:
            # Deal the flop
            self.community_cards.extend(self.deck.deal(3))
            self.phase = GamePhase.FLOP
        elif self.phase == GamePhase.FLOP:
            # Deal the turn
            self.community_cards.append(self.deck.deal())
            self.phase = GamePhase.TURN
        elif self.phase == GamePhase.TURN:
            # Deal the river
            self.community_cards.append(self.deck.deal())
            self.phase = GamePhase.RIVER
        elif self.phase == GamePhase.RIVER:
            # Go to showdown
            self.phase = GamePhase.SHOWDOWN
        
        # Reset bets for new round
        for player in self.players:
            player.bet = 0
            
        # Reset to first active player
        self.current_player = (self.dealer_position + 1) % self.num_players
        while (self.players[self.current_player].folded or 
               self.players[self.current_player].all_in):
            self.current_player = (self.current_player + 1) % self.num_players
    
    def _get_observation(self, player_idx: int) -> Dict[str, Any]:
        """
        Create observation for the specified player
        
        Args:
            player_idx: Index of the player to create observation for
            
        Returns:
            Dictionary with game state information from player's perspective
        """
        player = self.players[player_idx]
        
        # Create observation dictionary
        obs = {
            'hand': player.hand,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'stack': player.stack,
            'current_bet': player.bet,
            'phase': self.phase.value,
            'position': player_idx,
            'dealer_position': self.dealer_position,
            'num_players': self.num_players,
            'players_info': []
        }
        
        # Add information about other players (without revealing their cards)
        for i, p in enumerate(self.players):
            if i != player_idx:
                obs['players_info'].append({
                    'position': i,
                    'stack': p.stack,
                    'bet': p.bet,
                    'folded': p.folded,
                    'all_in': p.all_in
                })
        
        return obs