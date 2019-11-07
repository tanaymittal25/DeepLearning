import numpy as np

gamma  = 0.75 #Discount Factor
alpha = 0.9 #Learning Rate

#States
locationToState = {
    'L1': 0,
    'L2': 1,
    'L3': 2,
    'L4': 3,
    'L5': 4,
    'L6': 5,
    'L7': 6,
    'L8': 7,
    'L9': 8
}

#Actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

#Reward Table
rewards = np.array([
    [0,1,0,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0],
    [0,0,0,0,1,0,1,0,1],
    [0,0,0,0,0,0,0,1,0]
])

stateToLocation = dict((state, location) for location, state in locationToState.items())

def getOptimalRoute(start, end):
    
    #Copy Reward Matrix
    newRewards = np.copy(rewards)
    
    #Get End State
    endState = locationToState[end]
    
    #Set Goal State Reward
    newRewards[endState, endState] = 999
    
    #Q Algorithm
    Q = np.array(np.zeros([9,9]))
    
    for i in range(1000):
        
        #Pick a random state
        current = np.random.randint(0,9)
        
        #Traverse 
        playableActions = []
        
        for j in range(9):
            if newRewards[current, j] > 0:
                playableActions.append(j)
                
        #Pick an Action Randomly
        nextState = np.random.choice(playableActions)
        
        #Calculate Temporal Difference
        TD = newRewards[current, nextState] + gamma * Q[nextState, np.argmax(Q[nextState, ])] - Q[current, nextState]
        
        #Update Q Value using Bellman Equation
        Q[current, nextState] += alpha * TD
        
        # print('Epoch', i)
        # print(newRewards)
        # print(playableActions)
    
    #initialise Optimal Route    
    route = [start]
    
    #Start the route
    nextLocation = start
    
    # i = 0
    
    while nextLocation != end:
        
        startState = locationToState[start]
        
        #Fetch Next Highest
        nextState = np.argmax(Q[startState, ])
        
        nextLocation = stateToLocation[nextState]
        route.append(nextLocation)
        
        start = nextLocation
        
        # print('New Epoch', i)
        # i+=1
        # print(route)
        # print(start, startState, nextLocation)
    
    return route

print(getOptimalRoute('L9', 'L1'))
    