Here are a list of new features the website and agent will need to be designed for: 

## Count down mode
All players start with 5 cards and lose a card when their bluff is called or if they incorrectly call someone else's valid bid. 

A player with zero cards is eliminated. The game ends when only one player remains with cards. 
## High hand
If a player is given a bid of hand X, and the player believes this is infact the highest hand, then they are allowed to declare "highest hand" If this is the highest hand then the player who made the called bid is penalized a card and the player who successfully called high hand is rewarded. 

The penalty is losing a card in count down mode, and gaining a card in count up mode. The reward is gaining a card in count down mode and losing a card in count up mode. If the player loses a card in count down mode and only has 1 card, then the penalized player gains two cards. 

## Exact hand rule set
A second game mode needs to be added that requires bids to be made for exact hands in the pool. For instance, for a hand to be considered existent in the pool, a hand of 5 cards (or the entire pool of cards if the pool size is less than 5) must be made where the bid is the exact legal hand resulting in 5 cards.

For instance if the bid is pair of aces and the pool cards are Ace Ace Ace 2 3, then the bid is not valid as the only hand that can be made is a 3 of a kind. 
If instead the pool cards where Ace Ace Ace 2 3 4 then the pair call would be valid since the 5 card poker hand Ace Ace 2 3 4 can be formed which is a legal pair of aces in poker. 

## Rebuilding the blind and conditional agent. 

A new agent will have to be trained with montecarlo simulation for the exact rules variant. The old and new agents will be required to retrained for the high card rule as well. The count up or count down should not require a new rule set. 
The premise is the same, whenever a call has a less than 50% likelihood of existing, then it is reject.

The high hand will have to be accommodated for with the conditional bot, for instance in a 1v1 with 1 card each, if the player holds an Ace, they know very likely that they hold a high card.

## A secret easter egg, 5 kings in the deck. 
Allow a secret mode to be toggled on which puts 5 kings in the deck making 5 of a kind kings a valid hand which is higher ranked than all other hands. 