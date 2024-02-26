# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        if successorGameState.isWin(): 
            return 999999

        foodList = newFood.asList()
        
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDistances = [manhattanDistance(newPos, ghost) for ghost in ghostPositions]

        scoreDiff = successorGameState.getScore() - currentGameState.getScore()

        if action == Directions.STOP:
            scoreDiff -= 10


        if newPos in currentGameState.getCapsules():
            scoreDiff += 150 * len(currentGameState.getCapsules())

        numFoodLeft = len(foodList)
        numFoodLeftCurrent = len(currentGameState.getFood().asList())
        if numFoodLeft < numFoodLeftCurrent:
            scoreDiff += 200
        scoreDiff -= 10 * numFoodLeft

        scaredTimes = [ghost.scaredTimer for ghost in newGhostStates]
        ghostPositionDistances = [manhattanDistance(newPos, pos) for pos in ghostPositions]
        if sum(scaredTimes) > 0:
            ghostPositionDistances = [manhattanDistance(newPos, pos) for pos in ghostPositions]
            if min(ghostDistances) < min(ghostPositionDistances):
                scoreDiff += 200
            else:
                scoreDiff -= 100
        else:
            if min(ghostDistances) < min(ghostPositionDistances):
                scoreDiff -= 100
            else:
                scoreDiff += 200

        return scoreDiff

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max(minimax(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1) for action in state.getLegalActions(agentIndex))
            else:
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return min(minimax(state.generateSuccessor(agentIndex, action), depth, nextAgent) for action in state.getLegalActions(agentIndex))

        actions = gameState.getLegalActions(0)
        scores = [minimax(gameState.generateSuccessor(0, action), 0, 1) for action in actions]
        return actions[scores.index(max(scores))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alpha_beta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                v = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    v = max(v, alpha_beta(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:
                v = float("inf")
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                for action in state.getLegalActions(agentIndex):
                    v = min(v, alpha_beta(state.generateSuccessor(agentIndex, action), depth, nextAgent, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        actions = gameState.getLegalActions(0)
        alpha = float("-inf")
        beta = float("inf")
        scores = [alpha_beta(gameState.generateSuccessor(0, action), 0, 1, alpha, beta) for action in actions]
        return actions[scores.index(max(scores))]
        
    

    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max(expectimax(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1) for action in state.getLegalActions(agentIndex))
            else:
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return sum(expectimax(state.generateSuccessor(agentIndex, action), depth, nextAgent) for action in state.getLegalActions(agentIndex)) / len(state.getLegalActions(agentIndex))

        actions = gameState.getLegalActions(0)
        scores = [expectimax(gameState.generateSuccessor(0, action), 0, 1) for action in actions]
        return actions[scores.index(max(scores))]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: It uses an evaluation function to score game states for a pacman agent using search algorithms like minimax. 
    It calculates features like the score, the distance to the closest food, the distance to the closest ghost, the distance to the closest scared ghost, the number of remaining food, the number of remaining capsules, the number of remaining scared ghosts and the number of remaining normal ghosts. 
    It then calculates a total score using these features and returns it.
    Based on these features, it calculates a total score and returns it. The score is higher for game states favorable to Pacman. By maximizing this evaluation function over possible future states, the agent can pick optimal moves using adversarial search techniques like minimax.
    This basically acts as a heuristic function to guide game decisions by scoring board positions.
    """
    "*** YOUR CODE HERE ***"
    
    score = currentGameState.getScore()
    pacman_pos = currentGameState.getPacmanPosition()
    food_grid = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    food_distances = [manhattanDistance(pacman_pos, food_pos) for food_pos in food_grid.asList()]
    if len(food_distances) > 0:
        closest_food_distance = min(food_distances)
    else:
        closest_food_distance = 0

    ghost_distances = [manhattanDistance(pacman_pos, ghost_state.getPosition()) for ghost_state in ghost_states]
    if len(ghost_distances) > 0:
        closest_ghost_distance = min(ghost_distances)
    else:
        closest_ghost_distance = 0

    scared_ghost_distances = [manhattanDistance(pacman_pos, ghost_state.getPosition()) for ghost_state in ghost_states if ghost_state.scaredTimer > 0]
    if len(scared_ghost_distances) > 0:
        closest_scared_ghost_distance = min(scared_ghost_distances)
    else:
        closest_scared_ghost_distance = 0

    remaining_food = len(food_grid.asList())

    remaining_capsules = len(currentGameState.getCapsules())
    remaining_scared_ghosts = len([ghost_state for ghost_state in ghost_states if ghost_state.scaredTimer > 0])
    remaining_normal_ghosts = len([ghost_state for ghost_state in ghost_states if ghost_state.scaredTimer == 0])


    total_score = score - closest_food_distance - 2 * closest_ghost_distance - 3 * closest_scared_ghost_distance - 10 * remaining_food - 100 * remaining_capsules + 50 * remaining_scared_ghosts - 100 * remaining_normal_ghosts

    return total_score

# Abbreviation
better = betterEvaluationFunction
