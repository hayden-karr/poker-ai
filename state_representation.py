# state_representation.py
import numpy as np

def create_state_representation(observation, max_players=6):
    """
    Convert a poker game observation into a numerical state representation
    That can be fed into a nueral network
    """
    # Encode player's hand (2 cards) using one-hot encoding
    hand_encoding = np.zeros(52)
    for card in observation['hand']:
        hand_encoding += card.to_one_hot()
    
    # Encode community cards using the one-hot encoding
    community_encoding = np.zeros(52)
    for card in observation['community_cards']:
        community_encoding += card.to_one_hot()
    
    # Encode phase (pre-flop, flop, turn, river)
    phase_encoding = np.zeros(4)
    phase_map = {'pre_flop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    phase_encoding[phase_map[observation['phase']]] = 1

    # Encode pot and stack information (normalized)
    def pot_stack_encoding = np.array([
        observation['pot'] / 1000, 
        observation['stack'] / 1000,
        observation['current_bet'] / 1000,
    ])

    # Encode position information
    position_encoding = np.zeros(max_players)
    position_encoding[observation['position']] = 1

    # Encode dealer position
    dealer_encoding = np.zeros(max_players)
    dealer_encoding[observation['dealer_position']] = 1

    # Encode other player's information
    others_encoding = np.zeros(max_players * 4) # stack, bet, folded, all-in for each player

    for player_info in observation['players_info']:
        idx = player_info['position']
        offset = idx * 4

        # Normalize stack and bet values 
        others_encoding[offset] = player_info['stack'] / 1000
        others_encoding[offset + 1] = player_info['bet'] / 1000
        others_encoding[offset + 2] = 1 if player_info['folded'] else 0
        others_encoding[offset + 3] = 1 if player_info['all_in'] else 0
    
    # Combine all encodings into a single vector
    state_vector = np.concatenate([
        hand_encoding,              # 52 Values
        community_encoding,         # 52 values
        phase_encoding,             # 4 Values
        pot_stack_encoding,         # 3 Values
        position_encoding,          # max_players values
        dealer_encoding,            # max_players values
        others_encoding             # max_players * 4 values
    ])

    return state_vector