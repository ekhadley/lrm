# Do language models model expected rewards?
- total time: 13 hrs
### Summary
After pretraining, modern models are trained via some kind of RL using feedback datasets (winner/loser completion pairs, human likert ratings, ai ratings, etc). These teach them to produce generations that rate higher via the scoring process than the original model. Now there is also RLVR, aiming to make model's completions more likelt to satisfy some automatic grader for math/coding tasks.  
This project was chosen to get at a broad question: What tokens do models like to see in their inputs? The idea of what tokens models 'like' to produce is a clearer concept. Over the course of posttraining, the model produces responses that it estimates will receive higher reward from the grader. We can take a particular rollout and give it to both the pre-posttraining and post-posttraining model to see how much more likely the postraining made the model to generate that response. But what does this mean when we are considering inputs? Well, assuming the model has some estimate of its expected reward 

## goal
- models are trained via RL of various kinds to produce certain kinds of completions and not others. I hypothesize that models probably have some estimate of the expected reward from the rater they were trained for
    - tested via probing the trained on a feedback dataset
- models that have been trained via this process have a tendency to produce sequences that score higher via the rater's score
     - this should demonstrated via the probe's reward estimates being higher on average for the post-rl' model than the pre-rl model.
- if we  have a probe for estimating reward, we can use the model's distribution and search over possible prompts or continuations of partial prompts, to find those which the probe esetimates are high reward.
    - if the model really is biased to produce (to like?) sequences which the probe estimates will provide high reward, these are the sequences the model should 'like' to see.

## notes


- current two methods for obtaining reward (assuming the model was trained via DPO on winner/loser completion pairs)
    - we can train a probe on the model's activations to predict the 1-10 RLAIF ratings of the completion
        - this gives us an 'absolute reward'. One that is not invariant to a shift.
        - allows us to compare the rewawrd of two different completions to two different prompts.
            - Or just the prompts themselves
    - IF we have a pair of completions for the same prompt, we can find out how much more likely the posttrained model is to generate one completion vs the other.
        - this is just how DPO estimates the (shift invariant) reward.

- current best rating estimate probe hparams: linear probe, 24.resid_pre, lr=1e-4, bs=8, seq_pos=-1, wd=1e-4
    - I trained a nonlinear probe, it was not better than the probe and a bit slower so yeah. Sticking with linear.

- `if dpo is just teaching the model to produce sequences that get high reward, can we just use the model to directly generate user sequences to see which tokens it likes more?`
    - We would need a way of testing to see if the post-trained model's completions of the user prompt score higher on the DPO objective than the un-post-trained model's completions. 
    - how to get this if we don't have any dataset of 'prompt ratings' (as opposed to completion ratings which we do have datasets for)
        - The reward estimate probe would be another way to estimate this. making this method equivalent in a way to the 'do MCTS for rollouts that the probe likes' method.
    - we could also just sample rollouts and look for those where the difference in the prob of the rollout between the posttrained and base models is largest

- estimating reward from the prompt is really hard.
    - models do some planning ahead of time, but imagining what your entire response will look like or how it will score to the rater is a pretty deep lookahead


## todo
- sweep across layers and sequence positions
