import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from random import random as rand
import matplotlib.colors as mcolors
from utils import HSVToRGB

class Agent:

    def __init__(self, index, position, velocity, empathy=1, xenophobia=1, vision=200, 
                 dPosition=0, dSpeed=0, dEmpathy=0, dXenophobia=0, dVision=0, age=0, plotSize=20):
        
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
        self.plotSize=plotSize

        self.pltObj = plt.Circle(self.position, self.plotSize, color=self.color)
        
    
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
            
            herd_unit_velocity = herd_velocity

            self.velocity = np.linalg.norm(self.velocity)*(self.velocity + herd_unit_velocity)/np.linalg.norm(self.velocity + herd_unit_velocity)

    def herdVelocity(self, agents, distFactor=100):
        
        herd_velocity = np.zeros(2)

        for neighbor in agents:

            if neighbor.index is not self.index:

                distance = np.linalg.norm(neighbor.position - self.position)

                if distance < self.vision and distance > 0.1:
                    herd_velocity += neighbor.velocity * (0.5-abs(self.species-neighbor.species))  #*distFactor/distance

        return herd_velocity
    

    def updateEmpathy(self):
        return 0

    def updateXenophobia(self):
        return 0

    def updateVision(self):
        return 0