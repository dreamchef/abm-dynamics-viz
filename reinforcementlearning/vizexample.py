import gymnasium
import numpy as np
import copy

import random

GENS = 2
REPL = 2
CULL = 2
GRACE = 1 # avoid local minima?
EPISODES = 4

def get_all_keys(d, parent_keys=None):
    keys_list = []
    if parent_keys is None:
        parent_keys = []

    for key, value in d.items():
        current_keys = parent_keys + [key]
        keys_list.append(current_keys)
        if isinstance(value, dict):
            keys_list.extend(get_all_keys(value, current_keys))

    return keys_list

def modify_key_at_path(d, key_path, new_key):
    current_dict = d
    # Iterate until the second-to-last key in the path, to reach the parent dictionary of the key to be modified
    for key in key_path[:-1]:
        if key in current_dict:
            current_dict = current_dict[key]
        else:
            raise KeyError(f"Key {key} not found in the dictionary path.")

    # Replace the old key with the new key at the deepest level
    old_key = key_path[-1]
    if old_key in current_dict:
        current_dict[new_key] = current_dict.pop(old_key)
    else:
        raise KeyError(f"Key {old_key} not found in the dictionary.")

    return d  # Return the modified dictionary

# In this example, each policy takes the folloring form
#
#       engine_1 = form_1
#       engine_2 = form_2  
#       engine_3 = form_3
#       engine_4 = form_4
#
# where form_i is a function in terms of the observation space. If engine_i >= 0, it is on, else it is off (action potential).
# And form_1-4 are stored in the tree.

env = gymnasium.make('LunarLander-v2', render_mode='human')


def modify_key(d, key_path, new_key, current_level=0):

    key_path = key_path.split('.')

    if current_level == len(key_path) - 1:
        # At the target level, modify the key
        d[new_key] = d.pop(key_path[current_level])
    else:
        # Recurse deeper into the dictionary
        modify_key(d[key_path[current_level]], key_path, new_key, current_level + 1)


def policy_to_action(policy,obs):

    # print('yyy')
    #print(obs)
    # print('xxx')

    x = obs[0]
    y = obs[1]
    dx = obs[2]
    dy = obs[3]
    angle = obs[4]
    dangle = obs[5]
    L = obs[6]
    R = obs[7]

    print(x,y)

    L,R = bool(L),bool(R)

    return 0

def score_policy(policy):
    observation = env.reset()[0]  # Reset the environment to start a new episode
    total_reward = 0

    for episode in range(EPISODES):

        while True:
            # Render the environment (optional, can be slow)
            #env.render()

            # Take a random action (in this case, a random choice from the action space)

            #print('observation:',list(observation))

            action = policy_to_action(policy, list(observation))

            # Step the environment by applying the action
            observation, reward, done, info = env.step(action)[:4]

            #print(observation, reward, done, info)

            total_reward += reward

            if done:  # If the episode is finished
                break
    
    return total_reward/EPISODES



BIN_OPS = ['mult','div','add']
UN_OPS = ['neg','abs','exp','log','sq','sqrt','cb','sin','cos','d/dt','d2/dt2']
OPNDS = ['x','y','dx','dy','angle','dangle','L','R']

F = {'AP': {'x': '.'},
        'score': 0 # placeholder
        }

F['score'] = score_policy(F)



last_gen = [F]


def cull(batch):

    #print('...Batch: ',end=' ')

    for policy in batch[1:]:
        policy['score'] = score_policy(policy)

        #print(policy,end=' ')

    batch.sort(key=lambda x: x['score'], reverse=True)

    #print('Cull:',batch[:CULL],end=' ')

    return batch[:CULL]

def mutants(policy, sample=1):

    children = [policy]

    mutation_space = get_all_keys(policy['AP'])

    #print('Mutation space:',mutation_space)

    for i in range(REPL):

        mutation_target = random.choice(mutation_space)

        

        if len(mutation_space) > 1:
            mutation_space.remove(mutation_target)

        #print(mutation_target)

        new_policy = copy.deepcopy(policy)

        print('Mutating:',mutation_target,'in',new_policy['AP'],end=' ')

        if(mutation_target[-1] in BIN_OPS):
            new = random.choice(BIN_OPS)
            print('to',new)

            new_policy['AP'] = modify_key_at_path(new_policy['AP'],mutation_target,new)
        elif(mutation_target[-1] in UN_OPS):
            new = random.choice(UN_OPS)
            print('to',new)

            new_policy['AP'] = modify_key_at_path(new_policy['AP'],mutation_target,new)
        elif(mutation_target[-1] in OPNDS):
            new = random.choice(OPNDS)
            print('to',new)

            new_policy['AP'] = modify_key_at_path(new_policy['AP'],mutation_target,new)
        

        #modify_key(new_policy['AP'],mutation_target,random.choice(mutation_space))

        children.append(new_policy)

    # for mutation in mutations:
    #     children.append(mutation(policy))

    return children

for i in range(GENS):

    print('-'*100)
    print('\nGeneration',i,':',last_gen)

    next_gen = []

    for policy in last_gen:

        print('Evolving policies...')

        batch = cull(mutants(policy))

        print('\nSurvivors:',batch,'\n')

        for policy in batch:
            next_gen.append(policy) 
    
    last_gen = next_gen


print('\n\n')

# print out all the policies and their scores
last_gen.sort(key=lambda x: x['score'], reverse=True)
for policy in last_gen:
    print(policy['AP'])
    print(policy['score'])
    print('-'*20)
    print('\n')

env.close()  # Close the environment







    # # The tree is a dictionary where there is one key representing the operation and its value(s) are the operands
    # level = list(tree.items())[0]

#     for 

#     # Recursively compute the values of the operands
#     if isinstance(level, dict):  # Unary operation case (e.g., {'^2': 'y'})
#         operand_values = compute_formula(operands, values)
#     else:  # Binary operation case (e.g., {'+': {'x', 'y'}})
#         operand_values = [compute_formula(operand, values) for operand in operands]

#     # Apply the operation based on the key
#     if operation == '+':
#         return sum(operand_values)
#     elif operation == '-':
#         return operand_values[0] - operand_values[1]
#     elif operation == '*':
#         return operand_values[0] * operand_values[1]
#     elif operation == '/':
#         return operand_values[0] / operand_values[1]
#     elif operation == '^':
#         return operand_values[0] ** operand_values[1]
#     elif operation == '^2':  # Assuming '^2' means squaring the operand
#         return operand_values ** 2
#     else:
#         raise ValueError(f"Unsupported operation: {operation}")

# # Example usage
# formula_tree = {'+': [{'x': None}, {'^2': 'y'}]}  # Represents x + y^2
# values = {'x': 1, 'y': 3}
# result = compute_formula(formula_tree, values)
# print(f"Result: {result}")


#     print(observation)