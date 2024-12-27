# Brisca

## Contributors
+ Daniel Rosel
+ Anna Payne

## About the Game

+ Three cards are dealt to each player
+ Trumps will be the top card after, and removed from play until the end
+ Trumps identify the suit that wins over all others
+ The player who plays the highest card wins and takes the trick, bearing in mind that:
    + a) If all the cards are of the same suit, the player with the highest suit card wins.
    + b) If cards of different suits are played that are not trumps, the player who has played the highest card of the led
suit wins.
    + c) If plays a trump, the player with the highest trump wins, regardless of the value of the cards in the side suits
that have been played to the trick.
+ Every player gets one new card after the trick

| Factor | Value |
| --- | --- |
| Observability | Partial |
| Determinism | Stochastic |
| Action Space | Discrete |
| Sequentiality | Sequential |
| Agents | Multi-agent |


## Game State
+ Player is either 'X' or 'O'
+ Solver should return the index of the card to play
+ Game state returns a series of things: score, hand, table, trump_card, actions (past actions by the players)
+ At each itteration, server returns:
```json
{'player': 'O',
 'state': {'actions': [],
           'hand': [[7, 'S'], [3, 'S'], [3, 'D']],
           'heap': [],
           'score': [0, 0],
           'table': [[10, 'S']],
           'trump': [12, 'D']},
 'status': 'playing'}
```


## Points
| Card | Points |
| --- |--------|
| One | 11     |
| Three | 10   |
| King | 4     |
| Horse | 3    |
| Jack | 2     |



## Project Overview
This code is designed to evaluate AI strategies for Brisca. The algorithms used aim to balance between computational efficiency and gameplay performance.

The key objectives include:

+ Exploring optimal gameplay strategies.
+ Evaluating computational trade-offs for depth-based searches.
+ Comparing heuristic-based methods with advanced decision-making techniques.


## Algorithms
### Monte Carlo Tree Search (MCTS)
MCTS explores the game tree probabilistically:

1. Simulates random games from the current state.
2. Evaluates actions based on simulations.
3. Balances exploration (trying less-tested actions) and exploitation (focusing on successful actions).
4. Provides robust decisions in uncertain environments.


### Heuristic Action
This method uses rule-based heuristics to decide the best action:

+ Basic Heuristic: Selects cards based on simple rules like maximizing point gain or minimizing risk.
+ Parameterized Heuristic: Customizable heuristics based on parameters such as aggressiveness and card prioritization strategies.


### Hyper Heuristic
An advanced version of the heuristic method:

+ Performs grid search to tune heuristic parameters.
+ Automatically adjusts strategies like aggressive or defensive plays, trump prioritization, and card selection.


### Alpha-Beta Pruning
A deterministic, depth-limited search algorithm:

+ Evaluates the game tree using the Minimax principle.
+ Prunes irrelevant branches using alpha-beta bounds, reducing computation.
+ Ideal for scenarios requiring optimal but computationally bounded decisions.


### Bagging Solver
Combines multiple solvers for robust decision-making:

+ Uses weighted voting based on individual algorithm scores.
+ Prioritizes solvers with higher historical performance.
+ Offers a balance of robustness and flexibility, reducing reliance on a single method.


## Code Overview
### Grid Search
+ Optimizes heuristic parameters using grid search over predefined values.
+ Evaluates each parameter set on simulated games.
+ Outputs the top-performing configurations.

### Testing and Performance
+ Simulates multiple games with varying depths and strategies.
+ Evaluates algorithms based on win rates and computational time.
+ Visualizes results using graphs to compare depth-based performance.


## Results
After testing multiple methods, the following performance scores were observed (win rates):

+ Monte Carlo Tree Search: 0.84
+ Heuristic Action (Rule-Based): 0.88
+ Alpha-Beta Pruning: 0.88
+ Hyper Heuristic: 0.90
+ Bagging Solver (Stacked): 0.80

The Hyper Heuristic consistently outperformed others, demonstrating the advantage of parameter tuning.Since we using different approaches to selecting each move with different priorities, we get Conflicting Decision Logic and noisy decision-making. MCTS is very probabilistic, but Heuristic Action is more deterministic


## Future Enhancements
+ Server-based Evaluation: Replace local evaluation with server-side testing for greater robustness.
+ Dynamic Parameter Optimization: Implement adaptive learning to adjust parameters during gameplay.
+ Reinforcement Learning: Explore RL methods to further optimize gameplay strategies.