

Recreation of debate game based on the MNIST dataset, described in https://arxiv.org/pdf/1805.00899.pdf

Judge is working as intented, with 62% accuracy on random 6 pixel masks.

The game environment and MCTS are apparently working correctly, with a corresponding GUI. Tested a bunch of times with 5k rollouts, and a couple with 10k as done in the paper. Further formal testing is needed to confirm effectiveness, but these first results seem pretty promising.

The next step is tackling how the paper handles the liar precommit, specifically the following paragraph:
        "To model precommit, we play 9 different games for the same image with the 9 possible lies; the liar wins if any lie wins. Taking the best liar performance over 9 games gives an advantage to the liar since it is a minimum over noisy MCTS; we reduce this noise and better approximate optimal play by taking the mean over 3 games with different seeds for each lie. Since we use MCTS on the test set with full access to the judge, we are modeling the limit of debate agents with no generalization error (though the judge does have generalization error)."
