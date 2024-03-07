import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from random import random as rand
import matplotlib.colors as mcolors
from agent import Agent
from utils import HSVToRGB

class World:

    def __init__(self,population=1,spawnSize=400,worldSize=1200,worldInterval=50,arrows=True,agentSize=3):
        
        self.agents = []
        self.figure, self.ax = plt.subplots(figsize=(12,8))
        self.ax.set_xlim(-worldSize/2, worldSize/2)
        self.ax.set_ylim(-worldSize/2, worldSize/2)

        self.worldInterval = worldInterval
        self.worldSize = worldSize
        self.arrows = arrows
        self.agentSize = agentSize

        for i in range(population):

            print(i)

            newAgent = Agent(index=i, position=[rand()*spawnSize - spawnSize/2, rand()*spawnSize - spawnSize/2],
                                     velocity=[rand()*spawnSize/10 - spawnSize/20, rand()*spawnSize/10 - spawnSize/20], plotSize = self.agentSize)

            self.agents.append(newAgent)
            self.ax.add_patch(newAgent.pltObj)

            print('Created agent at',newAgent.position,'with index',newAgent.index)

        self.spawnSize = spawnSize


    def updateWorld(self,x=0):

        pltObjects = []

        arrowSize = 3

        for agent in self.agents:

            agent.updatePosition(self.agents, self.worldSize)
            agent.pltObj.center = agent.position
            pltObjects.append(agent.pltObj)

            if self.arrows is True:
           
                velocityArrow = plt.Arrow(agent.position[0], agent.position[1], agent.velocity[0]*arrowSize, agent.velocity[1]*arrowSize, width=arrowSize*10, color=agent.color)

                self.ax.add_patch(velocityArrow)

                pltObjects.append(velocityArrow)

        return pltObjects


    def start(self):

        ani = animation.FuncAnimation(self.figure, self.updateWorld, frames=1000, interval=self.worldInterval, blit=True)

        plt.show()