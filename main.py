import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

map = []

class Agent:
    def __init__(self, position, velocity, speed=0.1, empathy=1, xenophobia=1, vision=200, 
                 dPosition=0, dSpeed=0, dEmpathy=0, dXenophobia=0, dVision=0,
                species=0, age=0, ):
        self.velocity = np.array(velocity)
        self.speed = speed
        self.empathy = empathy
        self.xenophobia = xenophobia
        self.vision = vision
        self.empathy = empathy
        self.species = species
        self.position = np.array(position)
        self.dPosition = dPosition
        self.dSpeed = dSpeed
        self.dEmpathy = dEmpathy
        self.dXenophobia = dXenophobia
        self.dVision = dVision
        
    
    def updatePosition(self):

        self.updateVelocity()

        self.position += self.velocity

    def updateVelocity(self):

        herd_velocity = np.zeros(2)

        for neighbor in map:
            if np.linalg.norm(neighbor.position - self.position) < self.vision:
                herd_velocity += neighbor.velocity*self.empathy

        print(herd_velocity)

        self.velocity += (herd_velocity * self.speed / np.linalg.norm(herd_velocity) ) * 
    

    def updateEmpathy(self):
        return 0

    def updateXenophobia(self):
        return 0

    def updateVision(self):
        return 0
  
# add two agents to the map
[map.append(Agent([random.random()*200 - 100, random.random()*200 - 100], [random.random()*2 - 1, random.random()*2 - 1])) for i in range(5)]

# Set up the figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

agentMarkers = []

[agentMarkers.append(plt.Circle(map[i].position, 2, color=np.random.rand(3,))) for i in range(5)]
[ax.add_patch(agentMarkers[i]) for i in range(5)]

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
