# Snake-World-AI-Server
The AI server for [snake world](https://github.com/kwdChan/Snake-World/) using websocket. 
The AI doesn't do well yet.

## Debugging note
- 6Dec
  - Fixed rewards scheme: the model recieved rewards by repeatedly eating itself
  - Gamma was too high - getting longer means it's more likely to get hit in the future. The discounted punishments in the future need to be smaller than the rewards. Otherwise the model would decide not to eat.
  - Used elu instead of relu to prevent dead units
- 7Dec
  - The learning rate was okay: the predicted q values were improving over training steps.  
  - The rewards sometime does not work (likely the source of the problem)
  - The error was high when the model recieves rewards: the model does not expect the rewards at all.
    - try to selectively train on those steps to see if there's an error    

## 6 Dec 
The model is back to the spining mode and doesn't even eat the food. Starting to debug again. 


## 3 Dec 

Added enemy location. the model didn't behave well. 

Designing better inputs for the snake for the next one


## 2 Dec
**The model is doing something meaningful for the first time.**  

[Screencast from 02-12-23 19:11:46.webm](https://github.com/kwdChan/Snake-World-AI-Server/assets/64915487/c7df7d75-5ec5-4f2c-9751-c23da447bdae)

The input is very simple: the distance of the nearest food that is directly in the front, left and right to the snake and some information about the body locations. 

The model likes to spin the snakes when there's no food around. 
I guess it's because the snake can scan more area for foods. 

The debug should be done and it's time to add more complex inputs. 

## 1 Dec
Debug progress:

The rewards were given two steps after the action was taken. 

This was because the rewards were given according to the change in the length of the snake,
but the length of the snake changes only after the snake moves after it eats. 

This is detrimental to the learning as the model does not know it has eaten in the previous step and it would appear to the model that rewards are given randomly when there's no food. 


Also, the delay between Godot and Python can go above 100ms when there are too many snakes in the environment. 

If this is the case, the rewards for an action will only be received after another action is taken.




## 24 Nov 2023 model 2
Removed the length rewards and punishments for staying still
Same behaviour

Godot was frozen (likely due to the computer suspension) but the training continued in python with a replay buffer that wasn't large enough. 

Starting to investigate before training the next model


## 24 Nov 2023 model 1 - Step 30000
Removed some inputs (from 44 to 10)
Reduced the number of hidden units
Gave better inputs values

[Screencast from 24-11-23 13:23:36.webm](https://github.com/kwdChan/Snake-World-AI-Server/assets/64915487/90bd3838-3cef-486d-ac16-b80cabc0e527)

They learnt to keep spinning because small rewards were given constantly (proportional to the length) and punishments were given when the snake stays still


## model 23 Nov 2023
### Step 300000 
Same behaviour

### Step 5000 (very early)
I will let it sit for a night.  

[Screencast from 23-11-23 17:22:21.webm](https://github.com/kwdChan/Snake-World-AI-Server/assets/64915487/3e87c6fb-321f-4814-8b4f-34ea0c0aa860)
