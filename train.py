from texas_holdem import PokerGame, Action
from state_representation import create_state_representation
from poker_rl import PokerAgent

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def get_valid_actions(game, player_idx):
    """
    Determine which actions are valid for the current player
    """
    player = game.players[player_idx]
    current_max_bet = max(p['bet'] for p in game.players)
    call_amount = current_max_bet - player['bet']

    valid_actions = []

    # Fold is always an option unless checking is free
    if call_amount > 0:
        valid_actions.append(Action.Fold)
    
    # Check is valid if no additional betting required
    if call_amount == 0:
        valid_actions.append(Action.Check)
    
    # Call is valid if there is a bet to be called
    if call_amount > 0 and call_amount <= player['stack']:
        valid_actions.append(Action.Call)
    
    # Raise is valid if player has enough chips
    if player['stack'] > call_amount + 1:
        valid_actions.append(Action.Raise)
    
    # All in is always an option has at least a chip
    if player['stack'] > 0:
        valid_actions.append(Action.All_in)
    
    return valid_actions

def train_poker_agent(episodes=1000, evaluate_every=100):
    # Initialize poker game
    game = PokerGame(num_players=2)

    # initialize agents
    # for 2 players with a max of 6 , 5 actions!
    state_dim = 52 + 52 + 4 + 3 + 6 + 6 +(6 * 4)
    action_dim = 5 # fold, check, call, raise, all-in

    agent1 = PokerAgent(state_dim, action_dim)
    agent2 = PokerAgent(state_dim, action_dim)

    # Training metrics
    rewards_history = []
    win_rates = []

    # Initialize bet size tracking
    bet_states = []
    bet_sizes = []
    max_bets = []

    for episode in tqdm(range(episodes)):
        # Reset the environment 
        observation = game.reset()
        done = False

        # Initial state
        state = create_state_representation(observation)

        # Play until episode is done!
        while not done:
            # Get player index
            current_player = game.current_player

            # Get valid actions
            valid_actions = get_valid_actions(game, current_player)

            # Get action from currentagent
            if current_player == 0:
                action = agent1.get_action(state, valid_actions)
            else:
                action = agent2.get_action(state, valid_actions)
            
            # If action is Raise, determine bet size
            amount = 0
            if action == Action.Raise:
                player = game.players[current_player]
                current_max_bet = max(p['bet'] for p in game.players)
                max_possible_bet = player['stack'] + player['bet']

                if current_player == 0:
                    amount = agent1.get_bet_size(state, max_possible_bet)

                    #Track for training bet model
                    bet_states.append(state)
                    bet_sizes.append(amount)
                    max_bets.append(max_possible_bet)
                
                else:
                    amount = agent2.get_bet_size(state, max_possible_bet)
                
                #Ensure minimum raise
                amount = max(current_max_bet + 1, amount)
                # Ensure maximum raise
                amount = min(max_possible_bet, amount)
            
            # Take step in env
            next_observation, reward, done, info = game.step(action, amount)
            next_state = create_state_representation(next_observation)

            # Store Experience
            if current_player == 0:
                agent1.remember(state, action, reward, next_state, done)
            else:
                agent2.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state
            
            # Train agents
            if current_player == 0:
                agent1.replay()
            else:
                agent2.replay()

        # Train bet sizing model periodically
        if episode % 10 == 0 and bet_states:
            agent1.train_bet_model(bet_states, bet_sizes, max_bets)
            bet_states = []
            bet_sizes = []
            max_bets = []

        # Evaluate performance
        if (episode + 1) % evaluate_every == 0:
            win_rate = evaluate_agents(agent1, agent2, 100)
            win_rates.append(win_rate)

            print(f"Episode {episode+1}/{episodes}, Win Rate: {win_rate:.2f}") 

            # Save the model
            agent1.save_models(f"models/poker_agent_ep{episode +1}.h5")
        
        # Plot the traing progress
        plt.figure(figsize=(10,6))
        plt.plot(range(evaluate_every, episodes + 1, evaluate_every), win_rates)
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.title("Poker Agent Training Progress")
        plt.savefig('training_progress.png')
        plt.show()

        return agent1, agent2

def evaluate_agents(agent1, agent2, num_games=100):
    """
    Evaluate aget1's performance against agent 2
    """
    game = PokerGame(num_players=2)
    agent1_wins = 0

    for _ in range(num_games):
        observation = game.reset()
        done = False

        while not done:
            current_player = game.current_player
            state = create_state_representation(observation)
            valid_actions = get_valid_actions(game, current_player)

            if current_player == 0:
                action = agent1.get_action(state, valid_actions)
            else:
                action = agent2.get_action(state, valid_actions)

            amount = 0
            if action == Action.Raise:
                player = game.players[current_player]
                current_max_bet = max(p['bet'] for p in game.players)
                max_possible_bet = player['stack'] + player['bet']
            
                if current_player == 0:
                    amount = agent1.get_bet_size(state, max_possible_bet)
                else:
                    amount = agent2.get_bet_size(state, max_possible_bet)
            
                amount = max(current_max_bet + 1, amount)
                amount = min(max_possible_bet, amount)
            
            observation, reward, done, info = game.step(action, amount)

            if done and reward > 0 and current_player == 0:
                agent1_wins += 1
            
    return agent1_wins / num_games

if __name__ == "__main__":
    # Create directory for saved models
    os.makedirs("models", exist_ok=True)

    # Train agents
    agent1, agent2 = train_poker_agent(episodes=5000, evaluate_every=500)

    # final evaluation
    win_rate = evaluate_agents  (agent1, agent2, 1000)
    print(f"The final winrate is: {win_rate:.2f}")



