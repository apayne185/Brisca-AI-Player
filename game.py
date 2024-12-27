import random 
import json
import copy
from joblib import Parallel, delayed
from main import game_loop
from tqdm import tqdm



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

