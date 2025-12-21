# Notes
- total time: 0 hours
### Summary
Language models have been trained via some kind of rl on human preference data (or ai preference data). Language models thus are trained to steer their outputs to get high reward from the supervision process. (This used to be done with explicit reward models but now is done via direct optimization of the models, using their own knowledge as implicit reward models, but this isnt a load bearing fact.) Models most likely represent some form of their expected reward during generation. Can we try and elicit this information from the model? What effects the model's estsimate of reward? How do the user's messages factor in?

## goal
- to elicit from a chat model its estimate of its future reward from the reward modelling process
- to use this to figure out what completions the model thinks are good
- to use this to figure out what user inputs the model likes to see, as it expects its continuation to these inputs give high reward

## Status

## findings

## todo
