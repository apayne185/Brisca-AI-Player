# Brisca

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



