import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from random import random as rand
import matplotlib.colors as mcolors


def HSVToRGB(HSV):

    [h, s, v] = HSV

    if s == 0.0:
        return v, v, v
    i = int(h*6.)  # Assume H is given as a value between 0 and 1.
    f = (h*6.)-i
    p, q, t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f))
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

class World:

    def __init__(self,population=1,spawnSize=400,worldSize=1200,worldInterval=1000):
        
        self.agents = []
        self.figure, self.ax = plt.subplots(figsize=(18,10))
        self.ax.set_xlim(-worldSize/2, worldSize/2)
        self.ax.set_ylim(-worldSize/2, worldSize/2)

        self.worldInterval = worldInterval
        self.worldSize = worldSize
        
        for i in range(population):

            print(i)

            newAgent = Agent(index=i, position=[rand()*spawnSize - spawnSize/2, rand()*spawnSize - spawnSize/2],
                                     velocity=[rand()*spawnSize/10 - spawnSize/20, rand()*spawnSize/10 - spawnSize/20])

            self.agents.append(newAgent)
            self.ax.add_patch(newAgent.pltObj)

            print('Created agent at',newAgent.position,'with index',newAgent.index)

        self.spawnSize = spawnSize


    def updateWorld(self,x=0):

        pltObjects = []

        for agent in self.agents:

            agent.updatePosition(self.agents, self.worldSize)
            agent.pltObj.center = agent.position
            pltObjects.append(agent.pltObj)
           
            velocityArrow = plt.Arrow(agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1], width=2, color=agent.color)

            self.ax.add_patch(velocityArrow)

            pltObjects.append(velocityArrow)

        return pltObjects


    def start(self):

        ani = animation.FuncAnimation(self.figure, self.updateWorld, frames=100, interval=self.worldInterval, blit=True)

        plt.show()




class Agent:

    def __init__(self, index, position, velocity, empathy=1, xenophobia=1, vision=1200, 
                 dPosition=0, dSpeed=0, dEmpathy=0, dXenophobia=0, dVision=0, age=0, ):
        
        self.index = index
        self.velocity = np.array(velocity)
        self.empathy = empathy
        self.xenophobia = xenophobia
        self.vision = vision
        self.empathy = empathy
        self.species = rand()
        self.position = np.array(position)
        self.dPosition = dPosition
        self.dEmpathy = dEmpathy
        self.dXenophobia = dXenophobia
        self.dVision = dVision
        self.color = HSVToRGB([self.species,1,1])

        self.pltObj = plt.Circle(self.position, 5, color=self.color)
        
    
    def updatePosition(self, agents, worldSize):

        self.updateVelocity(agents)

        self.position += self.velocity

        if self.position[0] < -worldSize/2:
            self.position[0] += worldSize

        elif self.position[0] > worldSize/2:
            self.position[0] -= worldSize


        if self.position[1] < -worldSize/2:
            self.position[1] += worldSize

        elif self.position[1] > worldSize/2:
            self.position[1] -= worldSize

        # Update visualization objects
        self.pltObj.center = self.position
        


    def updateVelocity(self, agents):

        herd_velocity = self.herdVelocity(agents)

        herd_magnitude = np.linalg.norm(herd_velocity)
        self_magnitude = np.linalg.norm(self.velocity)

        if herd_magnitude > 0.1:
            
            herd_unit_velocity = herd_velocity/herd_magnitude

            self.velocity += herd_unit_velocity

    def herdVelocity(self, agents, distFactor=100):
        
        herd_velocity = np.zeros(2)

        for neighbor in agents:

            if neighbor.index is not self.index:

                distance = np.linalg.norm(neighbor.position - self.position)

                if distance < self.vision and distance > 0.1:
                    herd_velocity += neighbor.velocity * (1-np.sqrt(abs(self.species-neighbor.species)))  #*distFactor/distance

        return herd_velocity
    

    def updateEmpathy(self):
        return 0

    def updateXenophobia(self):
        return 0

    def updateVision(self):
        return 0


print('SETUP')

world = World(10)

print('\n\nSIMULATION')

world.start()
