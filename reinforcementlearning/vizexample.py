import gymnasium
import numpy as np
import copy

import random

GENS = 4
REPL = 8
CULL = 4
EPISODES = 7

P_B = [-0.7,0,0.6]

RECURSIVE_DEPTH = 0


import math

def policy_compute(policy, values):

    #print('passed policy:',policy)

    if isinstance(policy, str):
        if policy in values:
            #print('base case:',key,'=',values[key])
            return values[policy]
        else:
            print('ERROR')
        
    elif isinstance(policy, list):
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
            elif operation == 'sub':
                return left - right
            elif operation == 'mult':
                #print('multiplying',left,'by',right)
                if left is None or right is None:
                    print('ERROR: left:',left,'right:',right)
                return left * right
            elif operation == 'div':
                # Check for division by zero
                if right == 0:
                    return 0
                #print('dividing',left,'by',right)
                return left / right

        # Handle unary operations
        elif operation in UN_OPS:
            if len(branches) != 1:
                raise ValueError(f"Operation {operation} expects 1 operand, got {len(branches)}")

            operand_value = policy_compute(next(iter(branches)), values)
            
            if operation == 'abs':
                #print('abs  ')
                return abs(operand_value)
            elif operation == 'exp':
                #print('exp  ')
                return math.exp(operand_value)
            elif operation == 'logabs':
                #print('logabs  ')
                return math.log(abs(operand_value))
            elif operation == 'sin':
                #print('sin  ')
                return math.sin(operand_value)
            elif operation == 'cos':
                #print('cos  ')
                return math.cos(operand_value)
            elif operation == 'sqrtabs':
                #print('sqrtabs  ')
                return math.sqrt(abs(operand_value))
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
        

    else:
        print('ERROR')
        return 0
            
def potential_to_action(potential):

    if abs(potential-0) < 0.5:
        return 0
    
    elif abs(potential-0) < 1:
        return 2

    elif potential < 0:
        return 1
    
    else:
        return 3

    # if potential >= 0.6:
    #         return 3
    # elif potential >= 0:
    #         return 2
    # elif potential >= -0.7:
    #         return 1
    # else:
    #         return 0


def score_policy(policy, ep=10, render=False):
    observation = env.reset()[0]  # Reset the environment to start a new episode
    total_reward = 0

    sample = 0

    for episode in range(ep):

        #print('Episode:',episode)
        #print('-'*100)

        while True:
            # Render the environment (optional, can be slow)
            if render:
                env.render()

            # Take a random action (in this case, a random choice from the action space)

                
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

            #print('passing',policy['AP'])

            potential = policy_compute(policy['AP'], values)

            global RECURSIVE_DEPTH
            RECURSIVE_DEPTH = 0

            action = potential_to_action(potential)

            sample += 1

            if sample % 10 == 0:
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
        policy['score'] = score_policy(policy)

        #print(policy,end=' ')

    batch.sort(key=lambda x: x['score'], reverse=True)

    #print('Cull:',batch[:CULL],end=' ')

    return batch[:CULL]

def mutate_recursive(target):

    if isinstance(target, list):

        #print('mutating',target)

        random_element = random.choice(range(len(target)))

        #print('mutating',target[random_element])

        target[random_element] = mutate_recursive(target[random_element])

        return target

    else:

        #print('base mutating',target,'to',end=' ')

        if(target in BIN_OPS):
            new = random.choice(BIN_OPS)
            #print('to',new)

            #print(new)

            return new


        elif(target in UN_OPS):
            new = random.choice(UN_OPS)
            #print('to',new)

            #print(new)

            return new

        elif(target in OPNDS):
            new = random.choice(OPNDS)
            #print('to',new)

            #print(new)

            return new


def mutants(policy, sample=1):

    children = [policy]

    #print('Mutation space:',mutation_space)

    mutation_target = policy

    for i in range(REPL):

        new_policy = copy.deepcopy(policy)

        #print('planning to mutate',new_policy['AP'])

        new_policy['AP'] = mutate_recursive(new_policy['AP'])

        #print('birthing',new_policy)

        children.append(new_policy)

    return children




env = gymnasium.make('LunarLander-v2')#, render_mode='human')

BIN_OPS = ['mult','add','sub', 'div']
UN_OPS = ['abs','exp','log','sqrt','sin','cos']
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
last_gen.sort(key=lambda x: x['score'])

final_cull = last_gen [-30:]

for policy in final_cull:

    policy['score'] = score_policy(policy,ep=7)

final_cull.sort(key=lambda x: x['score'])

print('final popluation',len(last_gen))

for policy in final_cull:
    print(policy['AP'])
    print(policy['score'])
    print('-'*20)

print('final popluation',len(last_gen))

env.close()  # Close the environment