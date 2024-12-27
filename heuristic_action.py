# heuristic_action.py

import itertools
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from game import Brisca
from main import game_loop



def heuristic_action_basic(game_state):
    """
    Select the best card to play based on heuristics. this is the same as seen above in the MCTS
    Args:
        game_state: The current game state.
    Returns:
        Index of the best card to play from the hand.
    """
    game_state = game_state.game_state
    trump_suit = game_state.trump[1]  # Trump suit
    hand = game_state.state['hand']  # Player's hand
    table = game_state.state['table']  # Cards on the table

    # Helper to get card value
    def card_value(card):
        rank_values = {1: 11, 3: 10, 12: 4, 11: 3, 10: 2}  # Special values
        return rank_values.get(card[0], card[0])  # Default to face value

    # Split hand into trump and non-trump cards
    trump_cards = [card for card in hand if card[1] == trump_suit]
    non_trump_cards = [card for card in hand if card[1] != trump_suit]

    # if the difference between score is high, and we are losing, play the highest card
    # this is an aggressive strategy
    if game_state.state['score'][game_state.player] < game_state.state['score']["O" if game_state.player == "X" else "X"]:
        best_card = max(hand, key=card_value)
        return hand.index(best_card)

    # If there's a lead card on the table
    if table:
        lead_card = table[0]
        lead_suit = lead_card[1]

        # Find cards that match the lead suit
        lead_suit_cards = [card for card in hand if card[1] == lead_suit]

        # Determine if we need to win this round
        opponent_card = lead_card  # Only one card on the table so far
        must_win = card_value(opponent_card) > max(card_value(card) for card in hand) # if the opponent has a higher card than any of the cards in the hand

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
            best_trump = min(trump_cards, key=card_value)
            return hand.index(best_trump)

        # Otherwise, discard the lowest-value card
        return hand.index(min(non_trump_cards, key=card_value) if non_trump_cards else min(hand, key=card_value))

    # No lead card (you are leading), play the lowest-value card
    if trump_cards:
        worst_trump = min(trump_cards, key=card_value)
        return hand.index(worst_trump)
    return hand.index(min(hand, key=card_value))






def heuristic_action(game_state, params):
    """
    Select the best card to play based on parameterized heuristics.

    Args:
        game_state: The current game state.
        params: Dictionary of parameters to adjust heuristics.

    Returns:
        Index of the best card to play from the hand.
    """
    # Extract necessary elements from game_state
    game_state = game_state.game_state
    trump_suit = game_state.trump[1]   # The suit that is trump
    hand = game_state.state['hand']    # Player's hand
    table = game_state.state['table']  # Cards on the table
    player = game_state.player
    opponent = "O" if player == "X" else "X"


    def select_card(cards, strategy):
        # helper function to select a card based on strategy ('highest' or 'lowest')
        if not cards:
            return None
        return max(cards, key=card_value) if strategy == 'highest' else min(cards, key=card_value)


    def card_value(card):
        # helper function to get the value of a card
        rank_values = {1: 11, 3: 10, 12: 4, 11: 3, 10: 2}  # Special values in Brisca
        return rank_values.get(card[0], card[0])  # Default to face value if not special


        # Helper function to get cards with a higher value than a given value
    def get_higher_cards(cards, value):
        return [card for card in cards if card_value(card) > value]

    # Separate hand into trump and non-trump cards
    trump_cards = [card for card in hand if card[1] == trump_suit]
    non_trump_cards = [card for card in hand if card[1] != trump_suit]

    # Retrieve parameters with default values
    aggressive_threshold = params.get('aggressive_threshold', 0)
    aggressive_play = params.get('aggressive_play', 'highest')
    leading_play = params.get('leading_play', 'lowest')
    must_win_play = params.get('must_win_play', 'highest')
    cannot_win_play = params.get('cannot_win_play', 'lowest')
    trump_play = params.get('trump_play', 'lowest')
    discard_play = params.get('discard_play', 'lowest')

    # Calculate the score difference
    player_score = game_state.state['score'][player]
    opponent_score = game_state.state['score'][opponent]
    score_diff = opponent_score - player_score

    # Check if aggressive strategy should be applied
    if score_diff >= aggressive_threshold:
        # Aggressive play: select card based on aggressive_play strategy
        best_card = select_card(hand, aggressive_play)
        return hand.index(best_card)

    # If there is a lead card on the table (opponent has played)
    if table:
        lead_card = table[0]
        lead_suit = lead_card[1]
        opponent_card_value = card_value(lead_card)
        opponent_played_trump = (lead_suit == trump_suit)

        # Cards in hand that match the lead suit
        lead_suit_cards = [card for card in hand if card[1] == lead_suit]

        if opponent_played_trump:
            # Opponent played a trump card
            # Check if we have higher trump cards
            higher_trumps = get_higher_cards(trump_cards, opponent_card_value)
            if higher_trumps:
                # Play higher trump to win
                best_card = select_card(higher_trumps, must_win_play)
                return hand.index(best_card)
            else:
                # Cannot win; decide to play lead suit card or discard
                if lead_suit_cards:
                    best_card = select_card(lead_suit_cards, cannot_win_play)
                    return hand.index(best_card)
                else:
                    # Discard a card
                    discard_cards = non_trump_cards if non_trump_cards else hand
                    discard_card = select_card(discard_cards, discard_play)
                    return hand.index(discard_card)
        else:
            # Opponent did not play a trump
            # Check if we have higher cards in the lead suit
            higher_lead_suit_cards = get_higher_cards(lead_suit_cards, opponent_card_value)
            if higher_lead_suit_cards:
                # Play higher card in lead suit to win
                best_card = select_card(higher_lead_suit_cards, must_win_play)
                return hand.index(best_card)
            else:
                # Cannot win with lead suit cards
                if trump_cards:
                    # Play trump card to win
                    best_card = select_card(trump_cards, trump_play)
                    return hand.index(best_card)
                else:
                    # Cannot win; decide to play lead suit card or discard
                    if lead_suit_cards:
                        best_card = select_card(lead_suit_cards, cannot_win_play)
                        return hand.index(best_card)
                    else:
                        # Discard a card
                        discard_cards = non_trump_cards if non_trump_cards else hand
                        discard_card = select_card(discard_cards, discard_play)
                        return hand.index(discard_card)
    else:
        # No lead card (we are leading the trick)
        if leading_play == 'highest':
            best_card = select_card(hand, 'highest')
        elif leading_play == 'lowest':
            best_card = select_card(hand, 'lowest')
        elif leading_play == 'highest_trump':
            if trump_cards:
                best_card = select_card(trump_cards, 'highest')
            else:
                best_card = select_card(hand, 'highest')
        elif leading_play == 'lowest_trump':
            if trump_cards:
                best_card = select_card(trump_cards, 'lowest')
            else:
                best_card = select_card(hand, 'lowest')
        else:
            # Default to playing the lowest card
            best_card = select_card(hand, 'lowest')
        return hand.index(best_card)

    # Fallback: play the lowest card
    best_card = select_card(hand, 'lowest')
    return hand.index(best_card)







def grid_search(evaluate_strategy):
    """
    Perform a grid search over all possible combinations of heuristic parameters.
    """
    # Define possible values for each parameter
    # perform a grid search over all possible combinations of heuristic parameters.
    aggressive_threshold_values = [-5, 0, 5, 10]
    aggressive_play_values = ['highest', 'lowest']
    leading_play_values = ['highest', 'lowest', 'highest_trump', 'lowest_trump']
    must_win_play_values = ['highest', 'lowest']
    cannot_win_play_values = ['highest', 'lowest']
    trump_play_values = ['highest', 'lowest']
    discard_play_values = ['highest', 'lowest']

    # generate all combinations of parameter values
    parameter_grid = list(itertools.product(
        aggressive_threshold_values,
        aggressive_play_values,
        leading_play_values,
        must_win_play_values,
        cannot_win_play_values,
        trump_play_values,
        discard_play_values
    ))

    def evaluate_param_set(params):
        param_dict = {
            'aggressive_threshold': params[0],
            'aggressive_play': params[1],
            'leading_play': params[2],
            'must_win_play': params[3],
            'cannot_win_play': params[4],
            'trump_play': params[5],
            'discard_play': params[6]
        }
        # evaluate the strategy with the current parameter set
        performance_metric = evaluate_strategy(param_dict)
        return param_dict, performance_metric

    # Use joblib's Parallel and delayed to parallelize the grid search
    results = Parallel(n_jobs=-1)(delayed(evaluate_param_set)(params) for params in tqdm(parameter_grid, desc="Grid search"))

    # Optional: print the results if needed
    for param_dict, performance_metric in results:
        print(f"Tested parameters: {param_dict}, Performance: {performance_metric}")

    return results

def evaluate_strategy(params):
    # local evaluation agains random without game loop ortest_method
    # since we are doing it localy and our local implementation is slightly prone to error, the model resulting from this is possibly not the most optimal
    # an ideal grid-search would be against the server but sending that many requests has proven toilful and possibly costly
    game = Brisca({ "hand": [(1, 'H'), (3, 'D'), (10, 'S')], "score": { "X": 0, "O": 0 }, "table": [], "trump": (7, 'C') }, 'playing', 'X')
    while not game.is_end():
        action = heuristic_action(game, params)
        game.result(action)
    return game.game_state.state['score']['X']



grid_search_run = True
if not grid_search_run:
    print("Skipping grid search")
else:
    # Run the grid search
    results = grid_search(evaluate_strategy)

    # Find the best parameter set based on the performance metric
    best_params = max(results, key=lambda x: x[1])

    print(f"\nBest Parameters: {best_params[0]}, Best Performance: {best_params[1]}")



top_5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]
for params, performance in top_5:
    result = game_loop(lambda game: heuristic_action(game, params), Brisca, 'brisca', multi_player=False, id=None)
    print(f"Parameters: {params}, Performance: {performance}, Win rate: {result}")



ai_hyper = lambda game: heuristic_action(game, top_5[0][0])


    
       