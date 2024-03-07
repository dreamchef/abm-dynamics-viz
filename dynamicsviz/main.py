from world import World

print('SETUP')

world = World(population=100,agentSize=10)

print('\n\nSIMULATION')

world.start()