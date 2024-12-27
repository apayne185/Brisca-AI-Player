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
