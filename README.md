# Do language models model expected rewards?
- total time: 9 hrs
### Summary
Language models have been trained via some kind of rl on human preference data (or ai preference data). Language models thus are trained to steer their outputs to get high reward from the supervision process. (This used to be done with explicit reward models but now is done via direct optimization of the models, using their own knowledge as implicit reward models, but this isnt a load bearing fact.) Models most likely represent some form of their expected reward during generation. Can we try and elicit this information from the model? What effects the model's estimate of reward? How do the user's messages factor in?

## goal
- to elicit from a chat model its estimate of its future reward from the reward modelling process
- to see if the model has preferences over the tokens that it takes as input. This could mean a few things:
    - does the model simply represent an idea of the expected reward given the user's input?
    - does the model attempt to steer the conversation in ways that it expects will cause the user to produce inputs it likes?
        - is this why models when refusing will instead try to steer the user into doing something nicer instead and offering help with that?
        - would we expect this in a model that was not trained in an online fashion?

## notes

- is this model/dataset choice bad? (zephyr, a post train of mistral 1 7b from a gpt4-rated RLAIF completion dataset)
    - the labels are by gpt4, so pretty noisy
    - the model/dataset are around 2 years old
    - could we just... make a better model and dataset?
        - would involve getting a chat dataset,
        - rating each completion with some more modern model
            - 4o? sonnet 4.5? Not sure what the returns to scale are for something like this.
            - probably a gemini actually, they are pretty pareto optimal
        - training a chat model using this
            - sft? dpo? How exactly does the training scheme effect what the model is expected to learn or is incentivized to do?
                - I assume that whatever an AI completion rater is picking up on is also already represented linearly in the subject model and we are just promoting that direction or close to it during fting. so the probe can basically do the same thing?

    - since all i really care about is the raw ratings, I realy should be using the unbinarized dataset, rather than the binarized winner/loser dataset

- I trained a nonlinear probe, it was not better than the probe and a bit slower so yeah. Sticking with linear.

- training on layer 24 and with smaller batch size seems to work better.
    - Accuracy on par with previous ones, but the correlation is clearly much stronger on the scatterplot

## todo
- sweep across layers and sequence positions
- try lr scheduler

