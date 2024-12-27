import random
from heuristic_action import heuristic_action, heuristic_action_basic, ai_hyper
from minimax import MinimaxAlphaBetaPruning, alpha_beta
from mcts import montecarlo_solver, mcts
from game import test_method



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



def alpha_beta(game):
    solver = MinimaxAlphaBetaPruning(game)
    return solver.get_best_move()

def heuristic_action_basic(game):
    params = {
        'leading_play': 'highest',
        'must_win_play': 'highest',
        'cannot_win_play': 'lowest',
        'trump_play': 'highest',
        'discard_play': 'lowest',
        'aggressive_threshold': 0,
        'aggressive_play': 'lowest'
    }
    return heuristic_action(game.game_state, params)

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
