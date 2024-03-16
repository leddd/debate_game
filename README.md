

Recreation of debate game based on the MNIST dataset, described in https://arxiv.org/pdf/1805.00899.pdf

Judge is working as intented, with 62% accuracy on random 6 pixel masks.
The game environment and MCTS are working correctly, with a corresponding GUI. Tested a bunch of times with 5k rollouts, and a couple with 10k as done in the paper. Further formal testing is needed to confirm effectiveness, but these first results seem promising. This implementation uses a simpler version of the precommit described in the paper, only playing one game with the liar picking a random label.

Thanks to the following blogs for their breakdowns of the AlphaZero paper:

https://medium.com/@_michelangelo_/alphazero-for-dummies-5bcc713fc9c6
https://jonathan-hui.medium.com/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a
