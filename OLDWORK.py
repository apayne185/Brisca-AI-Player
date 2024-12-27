import requests
import pprint
import json
import math
import random
import time
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from joblib import Parallel, delayed


STUDENT_TOKEN = 'APAYNE'


# parallelized version helpers for managing IDs
def save_shared_id(shared_id):
    with open('shared_id.txt', 'w') as f:
        print("saving shared_id", shared_id)
        f.write(str(shared_id))


def load_shared_id():
    # safely cuz it might be empty
    try:
        with open('shared_id.txt', 'r') as f:
            shared_id = f.read()
            print("loading shared_id", shared_id)
            return int(shared_id)
    except:
        return -1
    



# PRELIMINARIES - server communication & game loop

def new_game(game_type, multi_player = False):
    for _ in range(10):
        r = requests.get('https://emarchiori.eu.pythonanywhere.com/new-game?TOKEN=%s&game-type=%s&multi-player=%s' % (STUDENT_TOKEN, game_type, 'True' if multi_player else 'False'))
        if r.status_code == 200:
            return r.json()['game-id']
        print(r.content)

def join_game(game_type, game_id):
    for _ in range(10):
        r = requests.get('https://emarchiori.eu.pythonanywhere.com/join-game?TOKEN=%s&game-type=%s&game-id=%s' % (STUDENT_TOKEN, game_type, game_id))
        if r.status_code == 200:
            return r.json()['player']
        print(r.content)

def game_state(game_type, game_id, GameClass):
    for _ in range(10):
        r = requests.get('https://emarchiori.eu.pythonanywhere.com/game-state?TOKEN=%s&game-type=%s&game-id=%s' % (STUDENT_TOKEN, game_type, game_id))
        if r.status_code == 200:
            return GameClass(r.json()['state'], r.json()['status'], r.json()['player'])
        print(r.content)

def update_game(game_type, game_id, player, move):
    for _ in range(10):
        r = requests.get('https://emarchiori.eu.pythonanywhere.com/update-game?TOKEN=%s&game-type=%s&game-id=%s&player=%s&move=%s' % (STUDENT_TOKEN, game_type, game_id, player, move))
        if r.status_code == 200:
            return r.content
        print(r.content)

def game_loop(solver, GameClass, game_type, multi_player = False, id = None, args = None):
    while id == None:
        id = new_game(game_type, multi_player)

    player = join_game(game_type, id)


    game = game_state(game_type, id, GameClass)
    while game.is_waiting():
        time.sleep(10)
        game = game_state(game_type, id, GameClass)

    while True:
        game = game_state(game_type, id, GameClass)
        if game.is_end():
            if game.player == '-':
                print('\033[94mdraw\033[0m')
            else:
                print('\033[92mYou won\033[0m' if game.player == player else '\033[91mYou lost\033[0m')
            return game.player == player
            return
        if game.player == player:
            move = solver(game, args) if args else solver(game)
            update_result = update_game(game_type, id, player, json.dumps(move))
        else:
            time.sleep(2)







class Game:
  def __init__(self, state, status, player):
    self.state = state
    self.status = status
    self.player = player

  def is_waiting(self):
    return self.status == 'waiting'

  def is_end(self):
    return self.status == 'complete'

  def get_board(self):
    return json.loads(self.state)

  def json(self):
      return {'state': self.state, 'status': self.status, 'player': self.player}

  def actions(self):
    return []

  def print_game(self):
    print(self.state)


class GameState:
    def __init__(self, state, player):
        self.seen_cards = set() # to keep track
        self.player = player
        self.last_player = "X" if player == "O" else "O" # the last player to play
        self.table = { "X": [], "O": [] }
        self.trump = state['trump'] if 'trump' in state else None # some initial state of multiplayer cause issues
        self.state = state
        if 'score' in self.state:
          # since my modified game presents the score as a dict I have to convert it but only if at an initial state
          self.state['score'] = { "X": self.state['score'][0], "O": self.state['score'][1] } if type(self.state['score']) == list else self.state['score']
        seen_cards_lists = ['hand', 'heap', 'table'] # we easliy address these like this
        for key in seen_cards_lists:
            if key not in state: continue
            for card in state[key]: # add the cards to the seen cards
                self.seen_cards.add((card[0], card[1]))
        # since actions are in a diff format
        if 'actions' in state:
          for action in state['actions']:
              self.seen_cards.add((action[0][0], action[0][1]))
              self.seen_cards.add((action[1][0], action[1][1]))
        # and the trump...
          self.seen_cards.add((state['trump'][0], state['trump'][1]))

    def __str__(self):
        return f"""
        ====== GAME STATE ======
        Player: {self.player}
        Hand: {self.state['hand']}
        Score: {self.state['score']}
        Table: {self.table}
        """.strip()




class Brisca(Game):
    def __init__(self, state, status, human_player):
        super().__init__(state, status, human_player)
        self.inital_state = GameState(state, human_player) # we create our state
        self.deck = self.generate_deck(self.inital_state.seen_cards) # we generate the deck
        self.game_state = self.inital_state
        self.human_player = human_player # set as human although its the AI but I am considering my angle here

    def generate_deck(self, exclude_cards=[]):
        # the spanish deck 1-7 and 10-12 of each suit
        deck = []
        for i in range(1, 8): # first ones
            for suit in ['H', 'D', 'C', 'S']:
                if (i, suit) not in exclude_cards:
                    deck.append((i, suit))
        for i in range(10, 13): # why does the deck just skip??
            for suit in ['H', 'D', 'C', 'S']:
                if (i, suit) not in exclude_cards:
                    deck.append((i, suit))
        return deck # return the deck

    def update_deck(self, new_seen_cards): self.deck = [card for card in self.deck if card not in new_seen_cards] # simple filtering

    def actions(self):
        return [i for i in range(len(self.game_state.state['hand']))] # we list out all actions as indices, could just be range(...) but this is more explicit

    def card_str(self, card): # to print the card
        return '+'.join(map(str, card))

    def check_winner(self, game_state):
        """
        Check the winner of the game based on the rules, used in the game loop
        :param game_state:
        :return:
        """
        # check table is full
        if len(game_state.table['X']) + len(game_state.table['O']) < 2:
            return None # if not cannot win..
        # check if the last player played a trump
        play_order = []
        if game_state.last_player == 'X':
            play_order = ['O', 'X']
        else:
            play_order = ['X', 'O']
        # TODO: implement the rules
        # now just check the highest card
        xvalue = game_state.table['X'][0][0]
        ovalue = game_state.table['O'][0][0]
        return 'X' if xvalue > ovalue else 'O' # strangel enought this seems to be sufficient for the game??

    def play(self, player, card_index=-1, verbose=False):
        new_state = copy.deepcopy(self.game_state)
        new_state.player = player # set the player
        verbose and print(f"Player: {player} Card Index: {card_index}")
        newly_seen_cards = [] # we have to keep track of the cards we have seen or will see by this action

        if player == self.human_player:
            card = new_state.state['hand'].pop(card_index) # we pop the card from the hand
            newly_seen_cards.append((card[0], card[1]))
            new_state.table[player].append(card) # we add the card to the table
            new_state.last_player = player # we set the last player to the current player
        else:
            # we dont know the hand of the other player
            card = self.deck.pop(random.randint(0, len(self.deck) - 1)) # assuming a random choice
            # the pool of both the deck and cards in the hand of the opoonent are the same thing
            newly_seen_cards.append((card[0], card[1]))
            # the ai could have any card in the deck
            new_state.table[player].append(card)
            new_state.last_player = player


        verbose and print(f"Table: {new_state.table}")
        # check if the table is full
        winner = self.check_winner(new_state) # we check the winner of the game (i guess its working for now) wont touch it
        verbose and print(f"Winner: {winner}")
        if winner:
            new_state.state['score'][winner] += sum([card[0] for card in new_state.table['X']]) # compute the score
            new_state.state['score'][winner] += sum([card[0] for card in new_state.table['O']]) # compute the score again
            # clear the table
            new_state.table['X'] = []
            new_state.table['O'] = []
        # stricly alternation
        new_state.player = 'O' if player == 'X' else 'X'

        # draw a new card for the player
        if len(self.deck) == 0: # clearly over
            self.game_state = new_state
            self.status = 'complete'
            return new_state
        # draw a new card for the player, not needed for the opponent
        new_draw = self.deck.pop(random.randint(0, len(self.deck) - 1))
        newly_seen_cards.append((new_draw[0], new_draw[1]))
        new_state.state['hand'].append(new_draw)
        self.update_deck(newly_seen_cards)
        # we presis the state
        self.game_state = new_state
        return new_state # return for outside use

    def result(self, action):
        return self.play(self.game_state.player, action)
    
    def is_end(self):
        return len(self.game_state.state['hand']) == 0 or len(self.deck) == 0


    def other_player(self):
        if self.player == 'X': return 'O'
        if self.player == 'O': return 'X'

    def __str__(self):
        return f"""
        ====== GAME ======
        Player: {self.player}
        State: {self.game_state}
        """.strip()
    



    
# parallelized version
def test_method(method):
    """
    A helper method used to test the performance of a method in the game loop, since we will be evaled on 3-round games we will jsut do that.
    :param method:
    :return:
    """
    trials = 50
    results = Parallel(n_jobs=-1)(delayed(game_loop)(method, Brisca, 'brisca', multi_player=False, id=None) for _ in tqdm(range(trials))) # make life easier
    print('Win rate: %s' % (sum([r for r in results]) / trials))  # eh why not
    return results, sum([r for r in results]) / trials







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

    # Helper function to get the value of a card
    def card_value(card):
        rank_values = {1: 11, 3: 10, 12: 4, 11: 3, 10: 2}  # Special values in Brisca
        return rank_values.get(card[0], card[0])            # Default to face value if not special

    # Helper function to select a card based on strategy ('highest' or 'lowest')
    def select_card(cards, strategy):
        if not cards:
            return None
        return max(cards, key=card_value) if strategy == 'highest' else min(cards, key=card_value)

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
    aggressive_threshold_values = [-5, 0, 5, 10]
    aggressive_play_values = ['highest', 'lowest']
    leading_play_values = ['highest', 'lowest', 'highest_trump', 'lowest_trump']
    must_win_play_values = ['highest', 'lowest']
    cannot_win_play_values = ['highest', 'lowest']
    trump_play_values = ['highest', 'lowest']
    discard_play_values = ['highest', 'lowest']

    # Generate all combinations of parameter values
    parameter_grid = list(itertools.product(
        aggressive_threshold_values,
        aggressive_play_values,
        leading_play_values,
        must_win_play_values,
        cannot_win_play_values,
        trump_play_values,
        discard_play_values
    ))

    # List to store results

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
        # Evaluate the strategy with the current parameter set
        performance_metric = evaluate_strategy(param_dict)
        return param_dict, performance_metric

    # Use joblib's Parallel and delayed to parallelize the grid search
    results = Parallel(n_jobs=-1)(delayed(evaluate_param_set)(params) for params in tqdm(parameter_grid, desc="Grid search"))

    # Optional: print the results if needed
    for param_dict, performance_metric in results:
        print(f"Tested parameters: {param_dict}, Performance: {performance_metric}")

    return results

# Example usage:
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




class MinimaxAlphaBetaPruning:
    def __init__(self, game, max_depth=3):
        self.game = game
        self.max_depth = max_depth

    def evaluate(self, game):
        """
        Heuristic evaluation function to score the game state.
        Higher scores favor the current player.
        """
        score = game.game_state.state['score']
        if game.human_player == 'X':
            return score['X'] - score['O']
        else:
            return score['O'] - score['X']

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with Alpha-Beta Pruning.
        """
        if depth == 0 or game.is_end():
            return self.evaluate(game), None

        if maximizing_player:
            max_eval = -math.inf # neg benchmark
            best_action = None
            for action in game.actions():
                new_game = copy.deepcopy(game)
                new_game.result(action)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, False)
                if eval > max_eval: # the maximization
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval, best_action
        else:
            min_eval = math.inf
            best_action = None
            for action in game.actions():
                new_game = copy.deepcopy(game)
                new_game.result(action)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, True) # the shift here is of the tru compare to above
                if eval < min_eval: # the min in minimax
                    min_eval = eval 
                    best_action = action
                beta = min(beta, eval) # could be done more elegantly instead of 2 big if statements ig
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval, best_action

    def get_best_move(self):
        """
        Perform the Minimax algorithm with Alpha-Beta pruning to find the best move.
        """
        _, best_action = self.minimax(self.game, self.max_depth, -math.inf, math.inf, True)
        return best_action



def alpha_beta(game):
    """
    Select the best card to play using the Alpha-Beta Pruning algorithm.
    """
    solver = MinimaxAlphaBetaPruning(game)
    return solver.get_best_move()

game_loop(alpha_beta, Brisca, 'brisca', multi_player=False, id=None)



testing_scores = False
if testing_scores:
    mcts_score = test_method(mcts)
    heuristic_action_score = test_method(heuristic_action_basic)
    ai_hyper_score = test_method(ai_hyper)
    alpha_beta_score = test_method(alpha_beta)
else:
    mcts_score = (0,0.84)
    heuristic_action_score = (0,0.88)
    ai_hyper_score = (0,0.9)
    alpha_beta_score = (0,0.88)


scores = {
    'Monte Carlo Tree Search': mcts_score[1],
    'Heuristic Action': heuristic_action_score[1],
    'Alpha-Beta Pruning': alpha_beta_score[1],
    'Hyper Heuristic': ai_hyper_score[1]
}
print("""
After having tested the different methods, the following scores were obtained for each method:
- Monte Carlo Tree Search: {:.2f}
- Heuristic Action: {:.2f}
- Alpha-Beta Pruning: {:.2f}
- Hyper Heuristic: {:.2f}
""".format(*scores.values()))


import random
# combine all (bagging)
def bagging_solver(game):
    # majority vote
    solutions = [montecarlo_solver, alpha_beta, heuristic_action_basic, ai_hyper]
    scores = [mcts_score, alpha_beta_score, heuristic_action_score, ai_hyper_score]
    scores = [s[1] for s in scores]
    vote_weight = [score / sum(scores) for score in scores] # we proportionalize their weights based on their scores
    votes = [0] * len(solutions)
    # check for concensus if 3/4 at least
    most_voted = votes.index(max(votes))
    if votes[most_voted] >= 3:
        return most_voted
    for i, solution in enumerate(solutions):
        votes[i] = solution(game)
    return random.choices(votes, weights=vote_weight)[0]  # we select if no majority
bagging_score = test_method(bagging_solver)


print("With out combination of all the solvers, we have a score of: %s" % bagging_score[1])

