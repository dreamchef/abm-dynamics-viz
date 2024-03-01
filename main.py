import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from random import random as rand

map = []

class World:

    def __init__(self,population=1,mapSize=400,):
        
        self.agents = []
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        
        for i in range(population):

            newAgent = Agent(position=[rand()*mapSize/10 - mapSize/20, rand()*mapSize/10 - mapSize/20],
                                     velocity=[rand()*mapSize/10 - mapSize/20, rand()*mapSize/10 - mapSize/20])

            self.agents.append(newAgent)
            self.ax.add_patch(newAgent.pltObj)

        self.mapSize = mapSize



class Agent:

    def __init__(self, position, velocity, empathy=1, xenophobia=1, vision=200, 
                 dPosition=0, dSpeed=0, dEmpathy=0, dXenophobia=0, dVision=0,
                species=0, age=0, ):
        self.velocity = np.array(velocity)
        self.empathy = empathy
        self.xenophobia = xenophobia
        self.vision = vision
        self.empathy = empathy
        self.species = species
        self.position = np.array(position)
        self.dPosition = dPosition
        self.dEmpathy = dEmpathy
        self.dXenophobia = dXenophobia
        self.dVision = dVision
        self.pltObj = plt.Circle(self.position, 2, color=np.random.rand(3,))
        
    
    def updatePosition(self):

        self.updateVelocity()

        self.position += self.velocity

    def updateVelocity(self):

        herd_velocity = np.zeros(2)

        for neighbor in map:
            if np.linalg.norm(neighbor.position - self.position) < self.vision:
                herd_velocity += neighbor.velocity*self.empathy

        print(herd_velocity)

        self.velocity += ((herd_velocity/np.linalg.norm(herd_velocity))-(self.velocity/np.linalg.norm(self.velocity)))
    

    def updateEmpathy(self):
        return 0

    def updateXenophobia(self):
        return 0

    def updateVision(self):
        return 0
  
# add two agents to the map
[map.append(Agent([random.random()*200 - 100, random.random()*200 - 100], [random.random()*2 - 1, random.random()*2 - 1])) for i in range(5)]

# Set up the figure and axis for animation


agentMarkers = []

[agentMarkers.append(plt.Circle(map[i].position, 2, color=np.random.rand(3,))) for i in range(5)]
[ for i in range(5)]

print(agentMarkers)

def update(frame):
    
    [agent.updatePosition() for agent in map]
    for i in range(5):
        agentMarkers[i].center = map[i].position 

    #print(map[0].velocity)

    return agentMarkers

# Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)

plt.show()
