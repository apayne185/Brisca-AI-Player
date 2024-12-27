import math
import random
from main import game_loop
from game import Brisca
import copy
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed



class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self, actions):
        return set(self.children.keys()) == set(actions)

    def best_child(self, exploration_weight=1.0):
        return max(
            self.children.items(),
            key=lambda child: child[1].wins / (child[1].visits + 1) +
                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (child[1].visits + 1))
        )[1] # this is for the UCB1 (Upper Confidence Bound) algorithm to select the best child i guess


def mcts_heuristic_action(actions, game_state): # late stage inspo post heuristic
    """
    Select the best card to play based on heuristics. Simple implementation.
    :param actions:
    :param game_state:
    :return:
    """


    # one could parameterize this and then do something like a grid search to find the best parameters for the heuristic
    # Helper to get card value
    def card_value(card):
        rank_values = {1: 11, 3: 10, 12: 4, 11: 3, 10: 2}  # Special values
        return rank_values.get(card[0], card[0])  # Default to face value

    game_state = game_state.game_state
    # Split hand into trump and non-trump cards
    trump_suit = game_state.trump[1]  # Trump suit
    hand = game_state.state['hand']  # Player's hand
    trump_cards = [card for card in hand if card[1] == trump_suit] # see any trump cards in our hand
    non_trump_cards = [card for card in hand if card[1] != trump_suit] # any non (could do sets but this is easier to understand)

    # If there's a lead card on the table
    if game_state.table[game_state.last_player]:
        lead_card = game_state.table[game_state.last_player][0] # let see what they played
        # maybe they play 3 of coins
        lead_suit = lead_card[1]  # ex: coins

        # our hand is 11 of coins, 7 of coins, 3 of cups (we are the player)
        # Find cards that match the lead suit
        lead_suit_cards = [card for card in hand if card[1] == lead_suit] # any matching?? # this would be 11 of coins and 7 of coins

        # Determine if we need to win this round
        opponent_card = lead_card  # Only one card on the table so far
        must_win = card_value(opponent_card) < max(card_value(card) for card in hand) # if the opponent has a higher card than any of the cards in the hand (no bueno)
        # ex: opponent played 3 of coins, we have 11 of coins and 7 of coins, we must win this round

        # Play the best lead suit card to win if possible
        if lead_suit_cards and must_win:
            best_card = max(lead_suit_cards, key=card_value)
            return hand.index(best_card)

        # Play the lowest card of the lead suit if winning isn't possible
        if lead_suit_cards:
            worst_card = min(lead_suit_cards, key=card_value)
            return hand.index(worst_card)

        # Play the lowest trump card to win
        if trump_cards:
            best_trump = min(trump_cards, key=card_value) # we play the lowest trump card to conserve the higher ones for later
            return hand.index(best_trump)

        # Otherwise, discard the lowest-value card
        return hand.index(min(non_trump_cards, key=card_value) if non_trump_cards else min(hand, key=card_value))

    # No lead card (you are leading), play the lowest-value card
    if trump_cards:
        worst_trump = min(trump_cards, key=card_value)
        return hand.index(worst_trump)
    return hand.index(min(hand, key=card_value))






def montecarlo_solver(game, iterations=1000):
    """
    Monte Carlo Tree Search to find the best action.
    Args:
        game: The current game state.
        iterations: Number of MCTS iterations to perform.
    Returns:
        The best action to play.
    """
    root = MCTSNode(game)
    print(game)

    for _ in range(iterations):
        node = root
        state_copy = copy.deepcopy(game)

        # Selection
        while node.is_fully_expanded(state_copy.actions()) and node.children:
            node = node.best_child()

        # Expansion
        actions = state_copy.actions()
        if not node.is_fully_expanded(actions):
            untried_actions = [a for a in actions if a not in node.children]
            action = random.choice(untried_actions) # we choose a random action from the untried actions
            state_copy.result(action)
            child_node = MCTSNode(copy.deepcopy(state_copy), parent=node)
            node.children[action] = child_node
            node = child_node

        # Simulation
        simulation_state = copy.deepcopy(state_copy)
        while not simulation_state.is_end():
            random_action = mcts_heuristic_action(simulation_state.actions(), simulation_state) # we could also just do random
            simulation_state.result(random_action) 
        winner = simulation_state.game_state.state['score'][game.player]

        # Backpropagation
        while node is not None:
            node.visits += 1
            if winner == game.player:
                node.wins += 1 # we increment the wins for our player
            node = node.parent

    # Return the action of the best child
    return max(root.children.items(), key=lambda item: item[1].wins / item[1].visits)[0] # we return the action with the highest wins to visits ratio
mcts = montecarlo_solver
game_loop(mcts, Brisca, 'brisca', multi_player=False, id=None)





depths = [100, 500, 1000] # 2000+ -> times out
times = [0] * len(depths)
wins = [0] * len(depths)

def run_game(depth):
    return 1 if game_loop(mcts, Brisca, 'brisca', multi_player=False, id=None, args=depth) else 0

run_tests = False
if not run_tests:
    print("Skipping tests")
else:
    for i, depth in enumerate(depths):
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_game, depth) for _ in range(3)]
            for future in as_completed(futures):
                wins[i] += future.result()
        times[i] = time.time() - start
        print(f"Depth: {depth}, Time: {times[i]:.2f} seconds, Wins: {wins[i]}")

    # visualize the results
    # show the time and wins as a function of depth time on left axis and wins on right axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(depths, times, 'g-')
    ax2.plot(depths, wins, 'b-')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Time', color='g')
    ax2.set_ylabel('Wins', color='b')
    plt.show()



