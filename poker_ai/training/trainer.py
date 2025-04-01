import tensorflow as tf
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import yaml
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('poker_training')


class ExperimentTracker:
    """Track experiment metrics and results"""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any], output_dir: str = "experiments"):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            output_dir: Directory to save experiment data
        """
        self.experiment_name = experiment_name
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            output_dir, f"{experiment_name}_{self.timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.metrics = {
            'episode_rewards': [],
            'win_rates': [],
            'policy_losses': [],
            'value_losses': [],
            'epsilon': []
        }
        
        # Save configuration
        with open(os.path.join(self.experiment_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Experiment {experiment_name} initialized at {self.experiment_dir}")
    
    def log_metrics(self, metrics: Dict[str, Any], episode: int) -> None:
        """
        Log metrics for an episode
        
        Args:
            metrics: Dictionary of metrics
            episode: Episode number
        """
        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Periodically save metrics
        if episode % 100 == 0:
            self.save_metrics()
            
    def save_metrics(self) -> None:
        """Save metrics to disk"""
        metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f)
    
    def plot_metrics(self) -> None:
        """Generate and save plots for tracked metrics"""
        # Win rate plot
        if self.metrics['win_rates']:
            plt.figure(figsize=(10, 6))
            win_rate_x = range(
                self.config['evaluate_every'], 
                len(self.metrics['win_rates']) * self.config['evaluate_every'] + 1, 
                self.config['evaluate_every']
            )
            plt.plot(win_rate_x, self.metrics['win_rates'])
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.title('Agent Win Rate vs. Episodes')
            plt.grid(True)
            plt.savefig(os.path.join(self.experiment_dir, 'win_rate.png'))
            plt.close()
        
        # Loss plots
        if self.metrics['policy_losses']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['policy_losses'], label='Policy Loss')
            plt.plot(self.metrics['value_losses'], label='Value Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.experiment_dir, 'losses.png'))
            plt.close()
        
        # Epsilon plot
        if self.metrics['epsilon']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['epsilon'])
            plt.xlabel('Training Step')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate (Epsilon)')
            plt.grid(True)
            plt.savefig(os.path.join(self.experiment_dir, 'epsilon.png'))
            plt.close()
    
    def save_agent(self, agent, checkpoint_name: str) -> None:
        """
        Save agent checkpoint
        
        Args:
            agent: Agent to save
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints', checkpoint_name)
        agent.save(checkpoint_dir)
        logger.info(f"Agent saved to {checkpoint_dir}")
    
    def get_checkpoint_dir(self, checkpoint_name: str) -> str:
        """
        Get path to a checkpoint directory
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Path to checkpoint directory
        """
        return os.path.join(self.experiment_dir, 'checkpoints', checkpoint_name)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_poker_agent(config: Dict[str, Any], experiment_tracker: Optional[ExperimentTracker] = None) -> Tuple:
    """
    Train poker agent with provided configuration
    
    Args:
        config: Training configuration
        experiment_tracker: Optional experiment tracker
        
    Returns:
        Tuple of trained agents
    """
    from environment.poker_env import PokerGame
    from models.state_encoder import create_state_representation
    from agents.poker_agent import PokerAgent, AgentConfig, ModelConfig
    
    # Extract configuration
    num_episodes = config.get('num_episodes', 5000)
    evaluate_every = config.get('evaluate_every', 500)
    save_every = config.get('save_every', 1000)
    
    # Initialize game environment
    game = PokerGame(
        num_players=config.get('num_players', 2),
        small_blind=config.get('small_blind', 1),
        big_blind=config.get('big_blind', 2),
        initial_stack=config.get('initial_stack', 100)
    )
    
    # Determine state and action dimensions
    state_dim = config.get('state_dim', 52 + 52 + 4 + 3 + 6 + 6 + (6 * 4))
    action_dim = config.get('action_dim', 5)
    
    # Create agent configurations
    model_config = ModelConfig(
        input_dim=state_dim,
        num_actions=action_dim,
        hidden_layers=config.get('hidden_layers', [256, 128]),
        dropout_rate=config.get('dropout_rate', 0.3),
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    agent_config = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=config.get('gamma', 0.95),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.1),
        epsilon_decay=config.get('epsilon_decay', 0.995),
        batch_size=config.get('batch_size', 64),
        buffer_size=config.get('buffer_size', 10000),
        model_config=model_config
    )
    
    # Initialize agents
    agent1 = PokerAgent(agent_config)
    agent2 = PokerAgent(agent_config)
    
    # If provided, load from checkpoints
    agent1_checkpoint = config.get('agent1_checkpoint')
    agent2_checkpoint = config.get('agent2_checkpoint')
    
    if agent1_checkpoint:
        agent1 = PokerAgent.load(agent1_checkpoint)
        logger.info(f"Loaded agent1 from {agent1_checkpoint}")
        
    if agent2_checkpoint:
        agent2 = PokerAgent.load(agent2_checkpoint)
        logger.info(f"Loaded agent2 from {agent2_checkpoint}")
    
    # Initialize bet size tracking
    bet_states = []
    bet_sizes = []
    max_bets = []
    
    # Training metrics
    episode_rewards = []
    
    # Progress bar
    progress_bar = tqdm(range(num_episodes), desc="Training")
    
    for episode in progress_bar:
        # Reset environment
        observation = game.reset()
        episode_reward = 0
        done = False
        
        # Initial state
        state = create_state_representation(observation)
        
        # Play until done
        while not done:
            # Get current player
            current_player = game.current_player
            
            # Get valid actions
            valid_actions = get_valid_actions(game, current_player)
            
            # Get action from current agent
            current_agent = agent1 if current_player == 0 else agent2
            action = current_agent.get_action(state, valid_actions)
            
            # Determine bet size if raising
            amount = 0
            if action.value == 3:  # Raise
                player = game.players[current_player]
                current_max_bet = max(p.bet for p in game.players)
                max_possible_bet = player.stack + player.bet
                
                # Get bet size from agent
                if current_player == 0:
                    amount = agent1.get_bet_size(state, max_possible_bet)
                    
                    # Track for training bet model
                    bet_states.append(state)
                    bet_sizes.append(amount)
                    max_bets.append(max_possible_bet)
                else:
                    amount = agent2.get_bet_size(state, max_possible_bet)
                
                # Ensure minimum and maximum bet
                amount = max(current_max_bet + 1, amount)
                amount = min(max_possible_bet, amount)
            
            # Take step in environment
            next_observation, reward, done, info = game.step(action, amount)
            next_state = create_state_representation(next_observation)
            
            # Store experience
            if current_player == 0:
                agent1.remember(state, action.value, reward, next_state, done)
                episode_reward += reward
            else:
                agent2.remember(state, action.value, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Train agents
            if current_player == 0 and len(agent1.memory) >= agent1.config.batch_size:
                metrics = agent1.train()
                
                # Log metrics if using experiment tracker
                if experiment_tracker:
                    for key, value in metrics.items():
                        if key not in experiment_tracker.metrics:
                            experiment_tracker.metrics[key] = []
                        experiment_tracker.metrics[key].append(value)
            
            elif current_player == 1 and len(agent2.memory) >= agent2.config.batch_size:
                agent2.train()
        
        # End of episode
        episode_rewards.append(episode_reward)
        
        # Train bet sizing model periodically
        if (episode + 1) % 10 == 0 and bet_states:
            agent1.train_bet_model(bet_states, bet_sizes, max_bets)
            bet_states = []
            bet_sizes = []
            max_bets = []
        
        # Evaluate performance periodically
        if (episode + 1) % evaluate_every == 0:
            win_rate = evaluate_agents(agent1, agent2, config.get('eval_episodes', 100))
            
            # Update progress bar
            progress_bar.set_postfix({
                'episode': episode + 1,
                'win_rate': f"{win_rate:.2f}",
                'avg_reward': f"{np.mean(episode_rewards[-100:]):.2f}"
            })
            
            # Log metrics
            if experiment_tracker:
                experiment_tracker.log_metrics(
                    {'win_rates': win_rate, 'episode_rewards': np.mean(episode_rewards[-100:])},
                    episode + 1
                )
                experiment_tracker.plot_metrics()
            
            logger.info(f"Episode {episode+1}/{num_episodes}, Win Rate: {win_rate:.2f}")
        
        # Save models periodically
        if (episode + 1) % save_every == 0:
            if experiment_tracker:
                experiment_tracker.save_agent(agent1, f"agent1_ep{episode+1}")
                experiment_tracker.save_agent(agent2, f"agent2_ep{episode+1}")
            else:
                # Create models directory if it doesn't exist
                os.makedirs('models', exist_ok=True)
                agent1.save(f"models/agent1_ep{episode+1}")
                agent2.save(f"models/agent2_ep{episode+1}")
    
    # Final evaluation
    final_win_rate = evaluate_agents(agent1, agent2, config.get('final_eval_episodes', 1000))
    logger.info(f"Final win rate: {final_win_rate:.4f}")
    
    # Final save
    if experiment_tracker:
        experiment_tracker.save_agent(agent1, "agent1_final")
        experiment_tracker.save_agent(agent2, "agent2_final")
        experiment_tracker.save_metrics()
        experiment_tracker.plot_metrics()
    else:
        os.makedirs('models', exist_ok=True)
        agent1.save("models/agent1_final")
        agent2.save("models/agent2_final")
    
    return agent1, agent2


def get_valid_actions(game, player_idx):
    """
    Determine valid actions for the current player
    
    Args:
        game: Poker game instance
        player_idx: Current player index
        
    Returns:
        List of valid Action enums
    """
    from environment.texas_holdem import Action
    
    player = game.players[player_idx]
    current_max_bet = max(p.bet for p in game.players)
    call_amount = current_max_bet - player.bet
    
    valid_actions = []
    
    # Fold is always an option unless checking is free
    if call_amount > 0:
        valid_actions.append(Action.FOLD)
    
    # Check is valid if no additional betting required
    if call_amount == 0:
        valid_actions.append(Action.CHECK)
    
    # Call is valid if there is a bet to call and player has enough chips
    if call_amount > 0 and call_amount <= player.stack:
        valid_actions.append(Action.CALL)
    
    # Raise is valid if player has enough chips
    if player.stack > call_amount + 1:
        valid_actions.append(Action.RAISE)
    
    # All-in is always an option if player has chips
    if player.stack > 0:
        valid_actions.append(Action.ALL_IN)
    
    return valid_actions


def evaluate_agents(agent1, agent2, num_games=100):
    """
    Evaluate agent1's performance against agent2
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        
    Returns:
        Win rate of agent1
    """
    from environment.poker_env import PokerGame
    from models.state_encoder import create_state_representation
    
    game = PokerGame(num_players=2)
    agent1_wins = 0
    
    for _ in range(num_games):
        observation = game.reset()
        done = False
        
        while not done:
            current_player = game.current_player
            state = create_state_representation(observation)
            valid_actions = get_valid_actions(game, current_player)
            
            # Get action from current agent
            if current_player == 0:
                action = agent1.get_action(state, valid_actions)
            else:
                action = agent2.get_action(state, valid_actions)
            
            # Determine bet size if raising
            amount = 0
            if action.value == 3:  # Raise
                player = game.players[current_player]
                current_max_bet = max(p.bet for p in game.players)
                max_possible_bet = player.stack + player.bet
                
                if current_player == 0:
                    amount = agent1.get_bet_size(state, max_possible_bet)
                else:
                    amount = agent2.get_bet_size(state, max_possible_bet)
                
                # Ensure minimum and maximum bet
                amount = max(current_max_bet + 1, amount)
                amount = min(max_possible_bet, amount)
            
            # Take step in environment
            observation, reward, done, info = game.step(action, amount)
            
            # Check if agent1 won
            if done and reward > 0 and current_player == 0:
                agent1_wins += 1
    
    return agent1_wins / num_games


def main():
    """Main entry point for training script"""
    parser = argparse.ArgumentParser(description='Train Poker AI')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default='poker_training',
                        help='Name for this experiment')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Directory to save experiment results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize experiment tracker
    experiment_tracker = ExperimentTracker(
        args.experiment_name,
        config,
        args.output_dir
    )
    
    # Train agent
    try:
        train_poker_agent(config, experiment_tracker)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save final metrics and plots
        experiment_tracker.save_metrics()
        experiment_tracker.plot_metrics()


if __name__ == "__main__":
    main()