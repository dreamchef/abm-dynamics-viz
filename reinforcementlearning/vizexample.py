import gymnasium
import numpy as np

# In this example, each policy takes the folloring form
#
#       engine_1 = form_1
#       engine_2 = form_2  
#       engine_3 = form_3
#       engine_4 = form_4
#
# where form_i is a function in terms of the observation space. If engine_i >= 0, it is on, else it is off (action potential)



def driver():

    env = gymnasium.make('LunarLander-v2', render_mode='human')

    GENS = 10
    REP = 5
    CULL = 1
    GRACE = 1 # avoid local minima

    BIN_OPS = ['mult','div','add']
    UN_OPS = ['neg','abs','exp','log','sq','sqrt','cb','sin','cos','d/dt','d2/dt2']
    

    env.close()  # Close the environment



def mutate_policy_to_children(policy, sample=1):

    children = []

    for mutation in mutations:
        children.append(mutation(policy))

    return children



def evaluate_policy(policy):
    observation = env.reset()  # Reset the environment to start a new episode
    total_reward = 0

    while True:
        # Render the environment (optional, can be slow)
        env.render()

        # Take a random action (in this case, a random choice from the action space)
        action = policy(observation)

        # Step the environment by applying the action
        observation, reward, done, info = env.step(action)[:4]

        total_reward += reward

        if done:  # If the episode is finished
            print("Episode {}: Total Reward: {}".format(episode + 1, total_reward))
            break






    




print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)

for episode in range(3):  # Run 3 episodes
    observation = env.reset()  # Reset the environment to start a new episode
    total_reward = 0

    while True:
        # Render the environment (optional, can be slow)
        env.render()

        # Take a random action (in this case, a random choice from the action space)
        action = env.action_space.sample()

        # Step the environment by applying the action
        observation, reward, done, info = env.step(action)[:4]

        total_reward += reward

        if done:  # If the episode is finished
            print("Episode {}: Total Reward: {}".format(episode + 1, total_reward))
            break

if __name__ == "__main__":
    driver()