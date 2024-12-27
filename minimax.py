import math
import copy
from main import game_loop
from game import Brisca

class MinimaxAlphaBetaPruning:
    def __init__(self, game, max_depth=3):
        self.game = game
        self.max_depth = max_depth

    def evaluate(self, game):
        """
        Heuristic evaluation function to score the game state.
        Higher scores favor the current player.
        """
        score = game.game_state['score']
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
            max_eval = -math.inf
            best_action = None
            for action in game.actions():
                new_game = copy.deepcopy(game)
                new_game.result(action)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = math.inf
            best_action = None
            for action in game.actions():
                new_game = copy.deepcopy(game)
                new_game.result(action)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def get_best_move(self):
        """
        Select the best card to play using the Alpha-Beta Pruning algorithm.
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

