import requests
# import pprint
import json
# import math
# import random
import time
# import copy
# from joblib import Parallel, delayed
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import itertools
# from joblib import Parallel, delayed
# from mcts import montecarlo_solver, mcts
# from minimax import alpha_beta
# from heuristic_action import heuristic_action_basic, ai_hyper
# from game import test_method


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




# testing_scores = False
# if testing_scores:
#     mcts_score = test_method(mcts)
#     heuristic_action_score = test_method(heuristic_action_basic)
#     ai_hyper_score = test_method(ai_hyper)
#     alpha_beta_score = test_method(alpha_beta)
# else:
#     mcts_score = (0,0.84)
#     heuristic_action_score = (0,0.88)
#     ai_hyper_score = (0,0.9)
#     alpha_beta_score = (0,0.88)


# scores = {
#     'Monte Carlo Tree Search': mcts_score[1],
#     'Heuristic Action': heuristic_action_score[1],
#     'Alpha-Beta Pruning': alpha_beta_score[1],
#     'Hyper Heuristic': ai_hyper_score[1]
# }
# print("""
# After having tested the different methods, the following scores were obtained for each method:
# - Monte Carlo Tree Search: {:.2f}
# - Heuristic Action: {:.2f}
# - Alpha-Beta Pruning: {:.2f}
# - Hyper Heuristic: {:.2f}
# """.format(*scores.values()))


# import random
# # combine all (bagging)
# def bagging_solver(game):
#     # majority vote
#     solutions = [montecarlo_solver, alpha_beta, heuristic_action_basic, ai_hyper]
#     scores = [mcts_score, alpha_beta_score, heuristic_action_score, ai_hyper_score]
#     scores = [s[1] for s in scores]
#     vote_weight = [score / sum(scores) for score in scores] # we proportionalize their weights based on their scores
#     votes = [0] * len(solutions)
#     # check for concensus if 3/4 at least
#     most_voted = votes.index(max(votes))
#     if votes[most_voted] >= 3:
#         return most_voted
#     for i, solution in enumerate(solutions):
#         votes[i] = solution(game)
#     return random.choices(votes, weights=vote_weight)[0]  # we select if no majority
# bagging_score = test_method(bagging_solver)


# print("With out combination of all the solvers, we have a score of: %s" % bagging_score[1])



