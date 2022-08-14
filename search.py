# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import queue
from cmath import inf
from itertools import accumulate
from queue import PriorityQueue
import util
from math import inf as INF


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# Please DO NOT change the following code, we will use it later
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, '',0, [])
    myPQ.push(startNode,heuristic(startState,problem))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()

        state, action, cost, path = node

        if (not state in visited) or cost < best_g.get(state):
            visited.add(state)
            best_g[state]=cost
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                myPQ.push(newNode,heuristic(succState,problem)+cost+succCost)
    util.raiseNotDefined()


def enforcedHillClimbing(problem, heuristic=nullHeuristic):
    """
    Local search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second argument (heuristic).
    """
    "*** YOUR CODE HERE FOR TASK 1 ***"

    # get start state
    startState = problem.getStartState()
    startNode = (startState, '', 0, [])

    # the parameter of startState is temporary, it will be changed in the loop
    while not problem.isGoalState(startState):
        # queue for all successors of the state
        myPQ = queue.Queue()
        # push the start node to the queue
        myPQ.put(startNode)
        # build the closed set
        visited = set()

        while not myPQ.empty():
            # get the node with the lowest cost (since it's a priority queue,
            # the node with the lowest cost is the first one in the queue)
            node = myPQ.get()
            # get the properties of the node
            state, action, cost, path = node
            # if the state is not in the closed set, add it to the closed set
            if not state in visited:
                visited.add(state)

                if heuristic(state, problem) < heuristic(startState, problem):
                    startState = state
                    startNode = (state, action, cost, path)
                    break

                for succ in problem.getSuccessors(state):
                    succState, succAction, succCost = succ
                    newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                    # we only care about the heuristic value of the successor as the priority value
                    myPQ.put(newNode)

    state, action, cost, path = startNode
    # if we find the goal state, return the path
    # get the actions from the path
    path = startNode[3]+[(state, action)]
    actions = [action[1] for action in path]
    # delete the first action, which is 'Stop'
    del actions[0]
    return actions

    
    # put the below line at the end of your code or remove it
    # util.raiseNotDefined()


from math import inf as INF   
def bidirectionalAStarEnhanced(problem, heuristic=nullHeuristic, backwardsHeuristic=nullHeuristic):
    
    """
    Bidirectional global search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call them.
    The heuristic functions are "manhattanHeuristic" and "backwardsManhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second and third arguments.
    You can call it by using: heuristic(state,problem) or backwardsHeuristic(state,problem)
    """
    "*** YOUR CODE HERE FOR TASK 2 ***"
    # The problem passed in going to be BidirectionalPositionSearchProblem
    # make initial node for each direction
    start_state = problem.getStartState()
    print("start_state:")
    print(start_state)
    start_node = (start_state, '', 0, [])
    # Since we have assuming there is only one goal state in assignment specification
    goal_state = problem.getGoalStates()[0]
    print("goal_state:")
    print(goal_state)
    goal_node = (goal_state, '', 0, [])

    Open_f = util.PriorityQueue()
    Open_b = util.PriorityQueue()

    # dfn = cost - hx_inverse_direction
    cost_f = start_node[2]
    print("backwardsHeuristic(start_state,problem):")
    print(backwardsHeuristic(start_state,problem))
    print("heuristic(start_state,problem):")
    print(heuristic(start_state,problem))
    dfn = cost_f - backwardsHeuristic(start_state, problem)
    print("dfn: ")
    print(dfn)
    Open_f.push(start_node, heuristic(start_state, problem)+cost_f+dfn)

    cost_b = goal_node[2]
    print("backwardsHeuristic(goal_state,problem):")
    print(backwardsHeuristic(goal_state,problem))
    print("heuristic(goal_state,problem):")
    print(heuristic(goal_state,problem))

    dbn = cost_b - heuristic(goal_state, problem)
    print("dbn: ")
    print(dbn)
    Open_b.push(goal_node, backwardsHeuristic(goal_state, problem)+cost_b+dbn)

    closed_f = set()
    closed_b = set()

    lower_bound = 0
    upper_bound = INF
    # x <- f, when x = true. x <- b, when x = false.
    x = True
    # no plan for now
    result = []

    while not Open_f.isEmpty() and not Open_b.isEmpty():
        print("U: ")
        print(upper_bound)
        print("L: ")
        print(lower_bound)

        # print("Open_f.heap: ")
        # print(Open_f.heap)
        # print("Open_b.heap: ")
        # print(Open_b.heap)
        b_Min_f = Open_f.getMinimumPriority()
        # print("b_Min_f: ")
        # print(b_Min_f)

        b_Min_b = Open_b.getMinimumPriority()
        # print("b_Min_b: ")
        # print(b_Min_b)
        lower_bound = (b_Min_b+b_Min_f)/2

        if x:
            n = Open_f.pop()
            # print("n:")
            # print(n)
            state, action, cost, path = n
            closed_f.add(state)
            # use directory to store Open_b
            directory_f = dict()
            directory_b = dict()
            for node_b in Open_b.heap:
                state1, action1, cost1, path1 = node_b[2]
                directory_f[state1] = (cost1, path1)
            for node_f in Open_f.heap:
                state1, action1, cost1, path1 = node_f[2]
                directory_b[state1] = (cost1, path1)

            if state in directory_b and cost + directory_b[state][0] < upper_bound:
                # print("directory_b[state][0]:")
                # print(directory_b[state][0])
                upper_bound = cost+directory_b[state][0]
                # print("directory_b[state][1]:")
                # print(directory_b[state][1])
                # print("path:///////")
                # print(path)
                del path[0]
                del directory_b[state][1][0]
                combine_path = path + directory_b[state][1]

                # print("combine_path_forward:")
                # print(combine_path)

                result = [action[1] for action in combine_path]


            if lower_bound >= upper_bound:
                return result

            successors = problem.getSuccessors(state)
            # successors.reverse()
            for succ in successors:
                # print("succ:")
                # print(succ)
                if succ[0] not in closed_f:
                    succState, succAction, succCost = succ
                    # print("succCost: -----------")
                    # print(succCost)
                    newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                    fn_f = heuristic(succState, problem)+cost+succCost
                    # print("heuristic(succState,problem):--------")
                    # print(heuristic(succState,problem))
                    # print("cost:--------")
                    # print(cost)
                    dn_f = cost + succCost - backwardsHeuristic(succState, problem)
                    bn_f = fn_f + dn_f
                    Open_f.push(newNode, bn_f)

        else:
            print("============backward===========")
            n = Open_b.pop()
            # print("n:")
            # print(n)
            state, action, cost, path = n
            closed_b.add(state)
            # use directory to store Open_b
            directory_f = dict()
            directory_b = dict()
            for node_b in Open_b.heap:
                state1, action1, cost1, path1 = node_b[2]
                directory_f[state1] = (cost1, path1)
            for node_f in Open_f.heap:
                state1, action1, cost1, path1 = node_f[2]
                directory_b[state1] = (cost1, path1)

            if state in directory_f and directory_f[state][0] + cost < upper_bound:
                # print("directory_f[state][0]:")
                # print(directory_f[state][0])
                upper_bound = directory_f[state][0] + cost

                del path[0]
                del directory_f[state][1][0]
                combine_path = path + directory_f[state][1]

                # print("combine_path:")
                # print(combine_path)

                result = [action[1] for action in combine_path]

            if lower_bound >= upper_bound:
                return result

            successors = problem.getBackwardsSuccessors(state)
            for succ in successors:
                # print("succ:")
                # print(succ)
                if succ[0] not in closed_b:
                    succState, succAction, succCost = succ
                    # print("succCost: -----------")
                    # print(succCost)
                    newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                    fn_b = heuristic(succState, problem) + cost + succCost
                    # print("heuristic(succState,problem):--------")
                    # print(heuristic(succState, problem))
                    # print("cost:--------")
                    # print(cost)
                    dn_b = cost + succCost - backwardsHeuristic(succState, problem)
                    bn_b = fn_b + dn_b
                    Open_b.push(newNode, bn_b)
        x = not x

    return result










    # put the below line at the end of your code or remove it
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


ehc = enforcedHillClimbing
bae = bidirectionalAStarEnhanced



