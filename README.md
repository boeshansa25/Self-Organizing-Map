"""
Aiden Boeshans
CSCI-252 Neuromorphic Computing: Lab 2 - Kohonen’s Self-Organizing Map
Due: September 27, 2024
Professor: Simon Levy
"""

import numpy as np
import matplotlib.pyplot as plt


class Som():
    
    def __init__(self, m, n = 2):
        self.m = m
        self.n = n
        self.som = np.random.random((m, m, n)) / 10 + 0.45
    
    def show(self, title="Self-Organizing Map"):
        """Plots the som on a grid with red points and blue lines by iterating
        through the som"""
        for j in range(self.m):
            #Plot the points
            for k in range(self.m):
                plt.plot(self.som[j,k,0], self.som[j,k,1], 'ro', markersize=4)
                for i in range(1, self.m-max(j,k)):
                #Plot the lines
                    plt.plot([self.som[j,k,0], self.som[j+1,k,0]], [self.som[j,k,1], self.som[j+1,k,1]], 'b-', markersize=2)
                    plt.plot([self.som[j,k,0], self.som[j,k+1,0]], [self.som[j,k,1], self.som[j,k+1,1]], 'b-', markersize=2)
        plt.gca().set_aspect('equal')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(data[:,0], data[:,1])
        plt.show()
        
    def _find_winner(self, e):
        """Finds the winning unit whose weights are closest to this e by iterating
        through the som and calculating their euclidean distance from e"""
        min_dist = np.inf
        for j in range(self.m):
            for k in range(self.m):
                dist = np.linalg.norm(e - self.som[j,k])
                if dist < min_dist:
                    min_dist = dist
                    winner = (j,k)
        return winner
    
    def _get_neighbors(self, p, d):
        """Return the neighbors of p within distance d"""
        neighbors = []
        for j in range(self.m):
            for k in range(self.m):
                if np.linalg.norm(np.array(p) - np.array((j,k))) < d:
                    neighbors.append((j,k))
        return neighbors

    def learn(self, data, T, a0, d0):
        """data: Training data, T: Number of iterations, α0: the initial learning rate, d0: 
        the initial neighborhood distance"""
        for t in range(T):
            # Compute Current Neighborhood Radius d and learning rate alpha
            d = d0 * (1-t/T)
            alpha = a0 * (1-t/T)
            
            # Pick an input "x" from the training set at random
            e = data[np.random.randint(len(data))]
            
            # Find the winning unit whose weights are closest to this e
            winner = self._find_winner(e)
            
            # Loop over the neighbors of this winner, adjusting their weights
            neighbors = self._get_neighbors(winner, d)
            for j, k in neighbors:
                self.som[j,k] += alpha * (e - self.som[j,k])
    

# Create Random Dataset
data = np.random.random((5000, 2))
r = (data[:, 0]-0.5)**2 + (data[:,1]-0.5)**2
data = data[np.logical_and(r>0.05, r<0.3)]
# Show the initial SOM
som = Som(10)
som.show('Initial SOM (Click x to continue)')
# Use dataset for training
som.learn(data, 100, 0.1, 5)
som.show("Trained SOM")
