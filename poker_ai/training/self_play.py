import numpy as np
import os
from poker_ai.agents.poker_agent import PokerAgent
from poker_ai.training.trainer import evaluate_agents

def self_play_training(iterations=10, games_per_iteration=1000):
    """
    Train agent through self-play against previous versions
    """
    # Initialize agent pool
    state_dim = 52 + 52 + 4 + 3 + 6 + 6 + (6 * 4)
    action_dim = 5

    # create main agent
    main_agent = PokerAgent(state_dim, action_dim)

    # create directory for saved models
    os.makedirs("self_play_models", exist_ok=True)

    # Save initial model
    main_agent.save_models(
        "self_play_models/agent_iter0_policy.h5",
        "self_play_models/agent_iter0_bet.h5"
    )

    # Agent pool (stode previous versions)
    agent_pool = [main_agent]

    for iteration in range(iterations):
        print(f"Self play iteration {iteration+1}/{iterations}")

        # Randomly select an opponent from pool where recent opponents are more likely
        weights = np.linspace(1, 10, len(agent_pool))
        weights = weights / np.sum(weights)
        opponent_idx = np.random.choice(len(agent_pool))
        opponent = agent_pool[opponent_idx]

        print(f"Selelcted opponent: {opponent_idx}")

        # Train main agent against the selected opponent
        from  poker_ai.training.trainer import train_poker_agent
        main_agent, _ = train_poker_agent(
            episodes=games_per_iteration,
            evaluate_every=games_per_iteration // 10,
            agent1 = main_agent,
            agent2 = opponent
        )

        # Evaluate against all previous agents
        total_win_rate = 0
        for i, old_agent in enumerate(agent_pool):
            win_rate = evaluate_agents(main_agent, old_agent, 100)
            print(f"Win rate against agent {i}: {win_rate:.2f}")
            total_win_rate += win_rate

        avg_win_rate = total_win_rate / len(agent_pool)
        print(f"Average win rate: {avg_win_rate:.2f}")

        #save the model
        main_agent.save_models(
            f"self_play_models/agent_iter{iteration+1}_policy.h5",
            f"self_play_models/agent_iter{iteration+1}_bet.h5"
        )

        # Create a copy of the agent for the pool
        new_agent = PokerAgent(state_dim, action_dim)
        new_agent.load_models(
            f"self_play_models/agent_iter{iteration+1}_policy.h5",
            f"self_play_models/agent_iter{iteration+1}_bet.h5"
        )

        agent_pool.append(new_agent)

    return main_agent