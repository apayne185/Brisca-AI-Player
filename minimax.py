import math
import copy
from main import game_loop
from game import Brisca, test_method
from mcts import montecarlo_solver, mcts
from minimax import alpha_beta
from heuristic_action import heuristic_action_basic, ai_hyper



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



