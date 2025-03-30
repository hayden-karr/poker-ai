#hand_evaluator.py
from collections import Counter

def evaluate_hand(hole_cards, community_cards):
    """
    Evaluate the best 5-card poker hand from the given hole cards and community cards
    returns a score that can be used to compare hands.
    Higher score is a better hand
    """
    all_cards = hole_cards + community_cards

    # Check for Royal flush
    royal_flush = check_royal_flush(all_cards)
    if royal_flush:
        return 9000000 + royal_flush # implement as if in the straight flush and return 1000000 extra?

    # Check for straight flush
    straight_flush = check_straight_flush(all_cards)
    if straight_flush:
        return 8000000 + straight_flush

    # Check for four of a kind
    four_kind = check_four_kind(all_cards)
    if four_kind:
        return 7000000 + four_kind
    
    # Check for full house
    full_house = check_full_house(all_cards)
    if full_house:
        return 6000000 + full_house
    
    # Check Flush
    flush = check_flush(all_cards)
    if flush:
        return 5000000 + flush
    
    # Check for straight
    straight = check_straight(all_cards)
    if straight:
        return 4000000 + straight
    
    # Check for 3 of kind
    three_kind = check_three_kind(all_cards)
    if three_kind:
        return 3000000 + three_kind
    
    # Check for two pair
    two_pair = check_two_pair(all_cards)
    if two_pair:
        return 2000000 + two_pair
    
    # Check for one pair
    one_pair = check_one_pair(all_cards)
    if one_pair:
        return 1000000 + one_pair
    
    # High Card
    return check_high_card(all_cards)

def check_straight_flush(cards)
    """
    Check for royal flush or straight flush
    """

    for suit in range(4):
        suited_cards = [card for card in cards if card.suit.value == suit]
        if len(suited_cards) >= 5:
            straight_value = check_straight(suited_cards)
            if straight_value:
                return straight_value
    
    return 0

def check_four_kind(cards):
    """
    Check for four of a kind
    """

    ranks = [card.rank for card in cards]
    counter = Counter(ranks)

    for rank, count in counter.items():
        if count >= 4:
            kickers = [r for r in ranks if r != rank]
            return rank * 100 + max(kickers)
    return 0

def check_full_house(cards):
    """
    Check for a full house
    """
    ranks = [card.rank for card in cards]
    counter = Counter(ranks)

    three_kind = None
    pair = None

    for rank, count in counter.most_common():
        if count >= 3 and three_kind is None:
            three_kind = rank
        if count count >= 2 and pair is None:
            pair = rank
    
    if three_kind and pair:
        return three_kind * 100 + pair
    
    return 0

def check_flush(cards):
    """
    Check for flush
    """
    for suit in range(4):
        suited_cards = [card for card in cards if card.suit.value == suit]
        if len(suited_cards) >= 5:
            ranks = sorted([card.rank for card in suited_cards], reverse=True)
            return sum(r * (100 ** i) for i,r in enumerate(ranks[:5]))
    return 0

def check_straight(cards):
    """
    Check for a straight
    """
    ranks = sorted(set(card.rank for card in cards), reverse=True)

    # Check for a A-5 straight
    if set([14,5,4,3,2]).issubset(set(ranks)):
        return 5 # 5-high straight

    # Check for regular straights
    for i in range(len(ranks)-4):
        if ranks[i] - ranks[i+4] ==4:
            return ranks[i] # Return highest card in straight
    
    return 0

def check_three_kind(cards):
    """
    Check for the three of a ind
    """
    ranks = [card.rank for card in cards]
    counter = Counter(ranks)

    for rank, count in counter.items():
        if count >= 3:
            kickers = sorted([r for r in ranks if r != rank], reverse =True)
            reutnr rank * 10000 + kickers[0] * 100 + kickers[1]
    return 0

def check_two_pair(cards):
    """
    Check for two pair
    """
    ranks = [card.rank for card in cards]
    counter = Counter(ranks)

    pairs = [rank for rank, count in counter.items() if count >= 2]
    if len(pairs) >= 2:
        pairs.sort(reverse=True)
        kickers = [r for r in ranks if r != pairs[0] and r != paris[1]]
        return pairs[0] * 10000 + pairs[1] * 100 + max(kickers)
    
    return 0

def check_one_pair(cards):
    """
    Check for one pair
    """
    ranks = [card.rank for card in cards]
    counter = Counter(ranks)

    for rank, count in counter.items():
        if count >= 2:
            kickers = sorted([r for r in ranks if r != rank], reverse=True)
            return rank * 1000000 + kickers[0] * 10000 + kickers[1] * 100 + kickers[2]
    return 0

def check_high_card(cards):
    """
    Evaluate a high card
    """
    ranks = sorted([card.rank for card in cards], reverse =True)
    return sum(r * (100 ** i) for i, r in emumerate(ranks[:5]))