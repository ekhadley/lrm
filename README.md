# Do language models model expected rewards?
- total time: 7 hrs
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
- Loaded starling 7b model which has publically released dataset and reward model
    - model sucks, using different model. Attempting to focus on DPO methods that require no explicit reward model
    - not sure this is possible:
        - dpo is used for contrastive sort of training. The reward you get is implicit, and only meainingful betweens pairs of completions for the same prompt
    - it is actually possible: the winners and losers were not chosen contrastively, but via [1-10] ratings from gpt4. The ratings are also provided, so those can be our source of 'absolute' rewards

- switching to 

- I have trained 1 probe on the layer 30 intial residual stream on the very last token position of the prompt+model response sequence.
    - it acheives a final rating accuracy (the percentage of the time that its guess is the same as the rating score from the dataset) of 35%
    - The dataset is balanced, so chance accuracy is 10%. So it's definitely learned something.
    - How good is this exactly?
        - Should make a scatterplot to see the variance

## todo
- make scatterplot of probe results to see correlation as well as accuracy
