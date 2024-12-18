# Project_04_SDS
## Project Description
This project is a simulation of driving swerve drive in FRC. I have been the backup driver for my FRC team for two years, and I can rarely find ways to practice without using the actual robot. Even with the actual robot, it is really difficult to practice offensive/defensive play. This simulation is an offensive/defensive drill, and uses box2d to simulate physics. I tried to write an AI to play this game using self play, however, it isn't very good right now. The goal of this game is for the blue robot to get past the red robot.

## Use of Game
### Setup
Install box2d, pygame, gymnasium and stable_baselines3 using pip
### Playing Game
Run Game/Main.py. Use WASD for red robot (Q and E for rotation), arrow keys for blue robot ([] for rotation).
## Self Play and Reinforcement Learning Structure
### Self Play
Self play consists of robots learning to play against each other. However, it is incredibly difficult to do so. You can't just train the two agents at the same time, because if one ends up dominating the other, the model dominating would learn behaviors against an easy opponent, while the other model wouldn't be able to find a solution to win. So, we have to add a structure to our reinforcement learning. I discovered that there are multiple types of self play-- asymmetric and symmetric. Asymmetric means that the agents who are playing against each other have different goals, while symmetric means the opposite.
### Structure
The structure I created(using the concepts from the youtube video below) is for asymmetric learning. However, it is important to understand some concepts from symmetric learning. In symmetric learning, to avoid playing against the same skill level and never learning, you create a model that gets trained in a policy network(PPO) normaly. Howvever, the opponent of this policy network is randomley from past networks. Let's say you finish a version of the network every 20 matches. The model will go against an opponent that is randomley selected from a list of 5 past networks, then after the model trains, it adds the new model to the list. This is the concept for symmetric learning. Asymmetric learning uses this concept, however, it just flips which agents its training. It will train blue against past versions of red, and red against past versions of blue, etc...
### PPO Model
I used a PPO model from stable_baselines3 with the following parameters for the red and blue robots:
```python
        self.blue_agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            verbose=1
        )
        
        self.red_agent = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            verbose=1
        )
```
### Reward System
The reward system that I implemented was:
#### Blue
- a small negative reward for time to make faster wins better.
- a positive reward at the end of the match for y position gained
- a +50 reward for winning and -50 for losing
#### Red
- a small positive or negative reward for distance to blue robot(closer=better)
- a small positive reward if red robot is in between blue robot and goal and negative if not
- +50 for winning, -50 for losing


## Improvements To Be Made
The reward system that I used was simple and it seems to work efficiently, however, my AI model does not work very well. I believe that it is because I haven't tuned the parameters correctly as once red learns a behavior, it tends to continue that behavior for quite some time.


## Sources
- https://bair.berkeley.edu/blog/2020/03/27/attacks/
- https://www.youtube.com/watch?v=0WMvzpAYJGk
- https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a
- I also used flint quite a lot in my implementation.

