

Recreation of the debate game based on the MNIST dataset, described in https://arxiv.org/pdf/1805.00899.pdf

Judge is working as intented, with 62% on random 6 pixel masks.

The game and agents are apparently working correctly, with a corresponding GUI. Tested a bunch of times with 5k rollouts, and a couple with 10k as done in the paper. Further formal testing is needed to confirm effectiveness and meassure judge accuracy, but these first results seems pretty promising; honest is winning a lot.