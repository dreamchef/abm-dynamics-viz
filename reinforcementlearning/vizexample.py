import gymnasium
import numpy as np
import copy

import random

GENS = 2
REPL = 2
CULL = 2
GRACE = 1 # avoid local minima?
EPISODES = 4

P_B = [0,2,4]

RECURSIVE_DEPTH = 0


import math

def policy_compute(policy, values):

    # Base case: the key is an operand
    if policy[0] in values:
        #print('base case:',key,'=',values[key])
        return values[policy[0]]

    # Recursive case: the key is an operation
    operation = policy[0]
    branches = policy[1:]
    #print('branches:',branches)

    # Handle binary operations
    if operation in BIN_OPS:

        if len(branches) != 2:
            raise ValueError(f"At {policy}, Operation {operation} expects 2 operands, got {len(branches)}")

        operands = [operand for operand in branches]

       #print(operands[0])
        #print(branches[operands[0]])
        
        left = policy_compute(operands[0], values)
        right = policy_compute(operands[1], values)

        if operation == 'add':
            return left + right
        elif operation == 'mult':
            #print('multiplying',left,'by',right)
            return left * right
        elif operation == 'div':
            # Check for division by zero
            if right == 0:
                raise ValueError("Division by zero")
            #print('dividing',left,'by',right)
            return left / right

    # Handle unary operations
    elif operation in UN_OPS:
        if len(branches) != 1:
            raise ValueError(f"Operation {operation} expects 1 operand, got {len(branches)}")

        operand_value = policy_compute(next(iter(branches)), values)

        if operation == 'neg':
            return -operand_value
        elif operation == 'abs':
            return abs(operand_value)
        elif operation == 'exp':
            return math.exp(operand_value)
        elif operation == 'log':
            if operand_value <= 0:
                raise ValueError("Log of non-positive number")
            return math.log(operand_value)
        elif operation == 'sin':
            return math.sin(operand_value)
        elif operation == 'cos':
            return math.cos(operand_value)
        elif operation == 'sqrt':
            if operand_value < 0:
                raise ValueError("Sqrt of negative number")
            return math.sqrt(operand_value)
    
    else:
        raise ValueError(f"Unknown operation: {operation}")
            
def potential_to_action(potential):

    if potential >= P_B[2]:
        return 3
    elif potential >= P_B[1]:
        return 2
    elif potential >= P_B[0]:
        return 1
    else:
        return 0
    


def score_policy(policy,render=False):
    observation = env.reset()[0]  # Reset the environment to start a new episode
    total_reward = 0

    sample = 0

    for episode in range(EPISODES):

        print('Episode:',episode)
        print('-'*100)

        while True:
            # Render the environment (optional, can be slow)
            if render:
                env.render()

            # Take a random action (in this case, a random choice from the action space)

            #print('observation:',list(observation))
                
            values = list(observation)
                
            values =    {'x': values[0],
            'y': values[1],
            'dx': values[2],
            'dy': values[3],
            'angle': values[4],
            'dangle': values[5],
            'L': values[6],
            'R': values[7]
            }

            potential = policy_compute(policy, values)

            global RECURSIVE_DEPTH
            RECURSIVE_DEPTH = 0

            action = potential_to_action(potential)

            sample += 1

            if sample % 100 == 0:
                print('observation',observation)
                print('potential',potential)
                print('action',action)

            # Step the environment by applying the action
            observation, reward, done, info = env.step(action)[:4]

            #print(observation, reward, done, info)

            total_reward += reward

            

            

            if done:  # If the episode is finished
                break
    
    return total_reward/EPISODES




def cull(batch):

    #print('Scoring next policy')

    for policy in batch[1:]:
        policy['score'] = score_policy(policy['AP'])

        #print(policy,end=' ')

    batch.sort(key=lambda x: x['score'], reverse=True)

    #print('Cull:',batch[:CULL],end=' ')

    return batch[:CULL]

def mutate_recursive(target):

    if isinstance(target, list):

        random_element = random.choice(range(len(target)))

        target[random_element] = mutate_recursive(target[random_element])

    else:

        if(target in BIN_OPS):
            new = random.choice(BIN_OPS)
            #print('to',new)

            return new


        elif(target in UN_OPS):
            new = random.choice(UN_OPS)
            #print('to',new)

            return new

        elif(target in OPNDS):
            new = random.choice(OPNDS)
            #print('to',new)

            return new


def mutants(policy, sample=1):

    children = [policy]

    #print('Mutation space:',mutation_space)

    mutation_target = policy

    for i in range(REPL):

        new_policy = copy.deepcopy(policy)

        new_policy['AP'] = mutate_recursive(new_policy['AP'])

        children.append(new_policy)

    return children




env = gymnasium.make('LunarLander-v2')#, render_mode='human')

BIN_OPS = ['mult','add']
UN_OPS = ['neg','abs','exp','log','sqrt','sin','cos']
OPNDS = ['x','y','dx','dy','angle','dangle','L','R']


# x*theta / (dx + dy)
F = {'AP': ['add', 
                ['mult','x','y'],
                ['mult','dx','dy']],
        'score': 0 # placeholder
        }

F['score'] = -200

print(F)

last_gen = [F]

print('start')

for i in range(GENS):

    print('-'*100)
    print('\nGeneration',i,':')

    for policy in last_gen:
        print(policy['AP'],'\n')

    next_gen = []

    for policy in last_gen:

        print('Evolving policies...')

        batch = cull(mutants(policy))

        print('\nSurvivors:\n')

        for policy in batch:
            print(policy['AP'],'\n')

        for policy in batch:
            next_gen.append(policy) 
    
    last_gen = next_gen


print('\n\n')



# print out all the policies and their scores
last_gen.sort(key=lambda x: x['score'], reverse=True)

score_policy(last_gen[0])


for policy in last_gen:
    print(policy['AP'])
    print(policy['score'])
    print('-'*20)
    print('\n')

env.close()  # Close the environment