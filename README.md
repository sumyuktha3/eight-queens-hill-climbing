# Hill Climbing Algorithm for Eight Queens Problem
## AIM

To develop a code to solve eight queens problem using the hill-climbing algorithm.

## THEORY
At each iteration, the hill-climbing search algorithm moves to the best successor of the current
node according to an objective function. The algorithm does not maintain a search tree, so the data structure for the current node need only
record the state and the value of the objective function.

## DESIGN STEPS

### STEP 1:
Import required python packages.


### STEP 2:
Define the Initial State and get the number of random conflicts at the initial, then using the objective function calculates.


### STEP 3:
Make a decision whether to change if a state with a better objective function value, or stay in the current state.
### STEP 4:
Repeat the process until the total number of conflicts, or the Objective function, becomes zero.
### STEP 5:
By calculating the time taken by the function to reduce the conflict for varying number of iterations.
### Step 6:
Plot a graph between time taken and iterations.


# PROGRAM
## Developed By: S.Sumyuktha Rani
## Register Number: 212220230050
~~~
%matplotlib inline
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
from IPython.display import display
from notebook import plot_NQueens
Problems
This is the abstract class. Specific problem domains will subclass this.
class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        
        raise NotImplementedError
    def result(self, state, action): 
        raise NotImplementedError
    def is_goal(self, state):        
        return state == self.goal
    def action_cost(self, s, a, s1): 
        return 1
    
    def __str__(self):
        return '{0}({1}, {2})'.format(
            type(self).__name__, self.initial, self.goal)
Nodes
This is the Node in the search tree. Helper functions (expand, path_actions, path_states) use this Node class.
class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __str__(self): 
        return '<{0}>'.format(self.state)
    def __len__(self): 
        return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): 
        return self.path_cost < other.path_cost
failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.
Helper functions
def expand(problem, state):
    return problem.actions(state)
Solving NQueens Problem using Hill Climbing
class NQueensProblem(Problem):

    def __init__(self, N):
        super().__init__(initial=tuple(random.randint(0,N-1) for _ in tuple(range(N))))
        self.N = N

    def actions(self, state):
        """ finds the nearest neighbors"""
        neighbors = []
        for i in range(self.N):
            for j in range(self.N):
                if j == state[i]:
                    continue
                s1 = list(state)
                s1[i]=j
                new_state = tuple(s1)
                yield Node(state=new_state)

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def goal_test(self, state):
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))

    def h(self, node):
        """Return number of conflicting queens for a given node"""
        num_conflicts = 0
        for (r1, c1) in enumerate(node.state):
            for(r2, c2) in enumerate(node.state):
                if (r1,c1)!=(r2,c2):
                    num_conflicts+= self.conflict(r1, c1,r2, c2)

        return num_conflicts
def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items
def argmin_random_tie(seq, key):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return min(shuffled(seq), key=key)
def hill_climbing(problem,iterations = 10000):
    # as this is a stochastic algorithm, we will set a cap on the number of iterations        
    current = Node(problem.initial)
    i=1
    while i < iterations:
        neighbors = expand(problem,current.state)
        if not neighbors:
            break
        neighbor = argmin_random_tie(neighbors,key=lambda node: problem.h(node))
        if problem.h(neighbor)<=problem.h(current):
            current.state= neighbor.state
            if problem.goal_test(current.state) == True:
                print("Goal test succeeded at iteration {0}.".format(i))
                return current
        i += 1        
    return current    
nq1=NQueensProblem(8)
plot_NQueens(nq1.initial)
n1 = Node(state=nq1.initial)
num_conflicts = nq1.h(n1)
print("Initial Conflicts = {0}".format(num_conflicts))
start=time.time()
sol1=hill_climbing(nq1,iterations=20000)
end=time.time()
print("Timetaken={0}".format(end-start))
sol1.state
num_conflicts = nq1.h(sol1)
print("Final Conflicts = {0}".format(num_conflicts))
plot_NQueens(list(sol1.state))
import time
iterations=[10,20,30,40,50,1000,2000,3000,4000,5000,10000]
timetaken=[]
num=1
for i in iterations:
    start=time.time()
    sol1=hill_climbing(nq1,iterations=i)
    end=time.time()
    print("The total time required for 2000 iterations is {0:.4f} seconds\n\n".format(end-start))
    timetaken.append(end-start)
    num+=1
import numpy as np
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
 
# Dataset
x = np.array(iterations)
y = np.array(timetaken)
 
X_Y_Spline = make_interp_spline(x, y)
 
# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
 
# Plotting the Graph
plt.plot(X_, Y_)
plt.title("graph between iteration and timetaken")
plt.xlabel("iterations")
plt.ylabel("timetaken")
plt.show()

 
# Dataset
x = np.array(iterations)
y = np.array(timetaken)
 
# Plotting the Graph
plt.plot(x, y)
plt.title("graph between x and y")
plt.xlabel("timetaken")
plt.ylabel("number of iteration")
plt.show()

~~~


## OUTPUT:
### Results.
![2](https://user-images.githubusercontent.com/75235032/169693877-e621b3c4-59db-41cb-8ce5-86e9ed1f2242.jpg)
### Shows the timetaken for each iterations.
![4](https://user-images.githubusercontent.com/75235032/169693889-de45258d-5479-4a63-8665-fdba523ddf78.jpg)

when the iterations increase, to complete the search time increases too.
### 8-Queen (initial conflicts).
![1](https://user-images.githubusercontent.com/75235032/169693907-27d4378e-d6ea-4c9b-b4a4-1b25f0411f52.jpg)
#### (final solution)
![3](https://user-images.githubusercontent.com/75235032/169693920-6c72f6a8-e7aa-44d7-9158-08dbe45927d7.jpg)




## Time Complexity Plot
#### Plot a graph for the various value of N and time(seconds)
![5](https://user-images.githubusercontent.com/75235032/169693942-5beca1ff-01b4-4f79-99e8-5279804481c2.jpg)
![6](https://user-images.githubusercontent.com/75235032/169693943-b0a16ecb-3d48-4f2c-938c-cfc73979275b.jpg)


## RESULT:
Hence, this code solves the eight queens problem using the hill-climbing algorithm that has been implemented.


