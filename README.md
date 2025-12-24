# Do language models model expected rewards?
- total time: 12 hrs
### Summary
Language models have been trained via some kind of rl on human preference data (or ai preference data). Language models thus are trained to steer their outputs to get high reward from the supervision process. (This used to be done with explicit reward models but now is done via direct optimization of the models, using their own knowledge as implicit reward models, but this isnt a load bearing fact.) Models most likely represent some form of their expected reward during generation. Can we try and elicit this information from the model? What effects the model's estimate of reward? How do the user's messages factor in?

## goal
- models are trained via RL of various kinds to produce certain kinds of completions and not others. I hypothesize that models probably have some estimate of the expected reward from the rater they were trained for
    - tested via probing the trained on a feedback dataset
- models that have been trained via this process have a tendency to produce sequences that score higher via the rater's score
     - this should demonstrated via the probe's reward estimates being higher on average for the post-rl' model than the pre-rl model.
- if we  have a probe for estimating reward, we can use the model's distribution and search over possible prompts or continuations of partial prompts, to find those which the probe esetimates are high reward.
    - if the model really is biased to produce (to like?) sequences which the probe estimates will provide high reward, these are the sequences the model should 'like' to see.

## notes

- current best hparams: linear probe, 24.resid_pre, lr=1e-4, bs=8, seq_pos=-1, wd=1e-4
    - I trained a nonlinear probe, it was not better than the probe and a bit slower so yeah. Sticking with linear.

- alternatively, we could train a probe on completions from the pre and post postraining model and have it calssify which model it thinks the completion came from.
    - `what would this tell us?`
    - a probe trained on the ground truth completion ratings tells us if the model contains enough info to estimate the things the rater cares about
        - this is potentially different from what the model *learns* to care about.
    - dpo is directly training, given winner/loser completion pairs, to maximize the difference between the model's likelihood to produce the losing completion and its likelihood to produce the winning one.

- agenda:
    - demonstrate that the trained model's completions receive higher reward according to the probe than the pre-rl model's completions
    - train the probe on different sequence positions in the assisant response, as well as at the very end of the user prompt, where no completion tokens are present.

- wait:
    - `if dpo is just teaching the model to produce sequences that get high reward, can we just use the model to directly generate user sequences to see which tokens it likes more?`
    - the way 

## todo
- sweep across layers and sequence positions
