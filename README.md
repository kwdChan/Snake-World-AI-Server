# Snake-World-AI-Server
The AI server for [snake world](https://github.com/kwdChan/Snake-World/) using websocket. 
The AI doesn't do well yet.


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
