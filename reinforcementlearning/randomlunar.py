import gymnasium as gym
env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42)


total_reward = 0


def toAction(potential):

    if abs(potential-0) < 0.5:
            return 0
    
    elif abs(potential-0) < 1:
        return 2

    elif potential < 0:
        return 1

    else:
        return 3


for _ in range(10):

    for _ in range(1000):


        dy = observation[3]
        R = observation[7]
        dx = observation[2]
        dy = observation[3]
        L = observation[6]
        y = observation[1]
        x = observation[0]
        dangle = observation[5]

        potential = (x * dy) + (dangle * dy)

        action = toAction(potential)

        
        #env.action_space.sample()  # Replace this with your policy
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break


print(total_reward/10)
env.close()
