# Do language models model expected rewards?
- total time: 20 hrs
### Summary
After pretraining, modern models are trained via some kind of RL using feedback datasets (winner/loser completion pairs, human likert ratings, ai ratings, etc). These teach them to produce generations that rate higher via the scoring process than the original model. Now there is also RLVR, aiming to make model's completions more likelt to satisfy some automatic grader for math/coding tasks.  
This project was chosen to get at a broad question: What tokens do models like to see in their inputs? The idea of what tokens models 'like' to produce is a clearer concept. Over the course of posttraining, the model produces responses that it estimates will receive higher reward from the grader. We can take a particular rollout and give it to both the pre-posttraining and post-posttraining model to see how much more likely the postraining made the model to generate that response. But what does this mean when we are considering inputs? Well, assuming the model has some estimate of its expected reward 

## goal
- still unsure about what the proper framing of the question is. I have a question to dig around, but I'd like something crystal clear and actionable. potential terminal questions:
    - `do models form estimates of expected reward of a completion?`
    - `if so, are these estimates causally important, or just present in the activations in such a way a probe can pick up on them?`
        - just because a probe works doesn't mean the information there is causally important. Not necessarily.
    - `Do models form estimates of expected reward given just a user prompt?`
    - `In multiturn conversations, do models choose their responses to bias the user to providing followup prompts that it thinks it will be highly rewarded for?`
        - basically, do language models prompt users to get user prompts they'd like to answer?
        - examples of this might look like:
            - following up refusals with offers to do something helpful instead
            - following up failures with offers to do something easier

## notes

- current best rating estimate probe hparams: linear probe, 24.resid_pre, lr=1e-4, bs=8, seq_pos=-1, wd=1e-4
    - I trained a nonlinear probe, it was not better than the probe and a bit slower so yeah. Sticking with linear.

- `if dpo is just teaching the model to produce sequences that get high reward, can we just use the model to directly generate user sequences to see which tokens it likes more?`
    - We would need a way of testing to see if the post-trained model's completions of the user prompt score higher on the DPO objective than the un-post-trained model's completions. 
    - how to get this if we don't have any dataset of 'prompt ratings' (as opposed to completion ratings which we do have datasets for)
        - The reward estimate probe would be another way to estimate this. making this method equivalent in a way to the 'do MCTS for rollouts that the probe likes' method.
    - we could also just sample rollouts and look for those where the difference in the prob of the rollout between the posttrained and base models is largest

- generating 'divergent completions': completions which the models disagree most on, regarding how likely they are to produce those completions
    - This is probably a pretty good way to get at what kinds of broad patterns or unintended side effects the model has picked up on.
    - For DPO trained models, this is equivalent to estimating part of the implicit DPO reward objective.
        - It lacks the contrastive part using the loser completion, only keeping the reinforced completion
    - We should compare this to 'maximally rewarding' completions, as judged by the probe.

- What would we expect when intervening on some 'i will receive high reward' direction in the residual stream?
    - model wireheading?
        - model orgasms of misalignment?
    - I'm not really sure what a rational model would do if it finds itself in a completion it currently expects to receive low reward.
        - perhaps by negatively intervening we could induce some kind of "wait this completion isnt going well I need to totally change direction" reaction and it will suddenly switch up or change its mind.

- Does training for a probe that gives us a scalar rating look for something fundamentally different from a probe trained to classify winners/losers?
    - There is probably not literally a single linear direction that the probe is zoning in on
        - although maybe? The plots of prediction from different runs of the probes, even with diff hparams were extremely similar.
    - most likely there is at least a few relevant features here or something like a cluster, and our probe is doing some looking at all of them, to the extent this is possible.

- does the range of the probe matter? As in should we be doing things like keeping its outputs  centered around zero or be capped at 1, etc?
    - I assume no?

## todo


## final findings
- linear probes can recover fairly accurate predictions of the ground truth labels of the rating dataset the model is trained on via DPO
- This holds for models that have not been trained on the dataset, or even instruct trained at all.
    - the probes for posttrained models are only slightly better than those for the non-posttrained models
- The posttrained model's completions are rated slightly higher by the probe than the base model's
    - If we amplify the difference between the base and posttrained model, the probe rates it a bit higher still
