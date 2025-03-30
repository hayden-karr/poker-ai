# poker_env.py
import numpy as np 
from enum import Enum 

class Suit(Enum):
    Hearts = 0
    Diamonds = 1
    Clubs = 2
    Spades = 3

class Card:
    def __init__(self, rank, suit):
        """
        Initialize a card with a rank (2-14, where 11-14 are face cards) and suit
        """
        self.rank = rank
        self.suit = suit
    
    def __str__(self):
        rank_str = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(self.rank, str(self.rank))
        suit_str = {Suit.Hearts: '❤️', Suit.Diamonds: '♦', Suit.Clubs: '♧', Suit.Spades: '♤'}[self.suit]
        return f"{rank_str}{suit_str}"
    
    def to_one_hot(self):
        """
        Convert card to one-hot encoding (52-dimensional vector)
        """
        one_hot = np.zeros(52)
        idx = self.suit.value * 13 + (self.rank-2)
        one_hot[idx] = 1
        return one_hot
    
class Deck:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        Create a fresh, ordered deck of 52 cards
        """
        self.cards = []
        for suit in Suit:
            for rank in range(2, 15): # 2 through Ace (14)
                self.cards.append(Card(rank,suit))
        self.shuffle()

    def shuffle(self):
        """
        Shuffle the deck
        """
        np.random.shuffle(self.cards)

    def deal(self, n=1):
        """
        Deal n cards from the deck
        """
        if n > len(self.cards):
            raise ValueError("Not enough cards left in the deck")
        dealt_cards = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt_cards if n > 1 else dealt_cards[0]