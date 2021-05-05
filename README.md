# Q-learning-to-solve-OpenAI-Gyms-Taxi-v3

This project focuses on using Q-learning, a temporal difference learning approach to solving an OpenAI-Gyms Taxi-v2 task.

The monte carlo control method was not sufficient to finding an optimal solution for this game. TD learning enabled the agent to learn after each timestep, instead of waiting to play an entire game (episode). There are many variations to the TD Learning Sarsa algorithms, and this particular projet uses Sarsamax (a.k.a Q Learning).

" This task was introduced in [Dietterich2000] to illustrate some issues in hierarchical reinforcement learning. There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions. " 

Please see https://gym.openai.com/envs/Taxi-v2/ for more information.


