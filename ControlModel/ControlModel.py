from numpy import *
import TrainingData

class ControlModel:
    # Constants
    WAIT_PENALTY = -0.000001
    CORRECT_REWARD = 10
    WRONG_PENALTY = -30
    LEARNING_RATE = 0.2
    TIME_DISCOUNT = 0.99
    STRATA_PER_CLASS = 7
    OBS_CARDINALITY = 15
    BOLTZMANN_TEMPERATURE = 1.3
    
    def __init__(self, trainingFile):
        _CalculateObservationPosteriors(trainingFile)
        
        # Define the actions and states of the POMDP
        # self.Actions = ['yes', 'wait', 'no']
        # self.States = ['yes', 'wait', 'no']
        
        # Each state is equally likely at first (uniform prior)
        # self.belief = ones(len(self.States)) / len(self.States)
        
        # ~~~ Not sure what this does yet
        # Reinforcement Learning reward function
        #   Incentivize initial exploration by setting ExpectedReward[Control, State] above the wait penalty
        self.ExpectedRewards  = -3 * ones((len(self.Actions), len(self.States))) 
        self.ExpectedRewards +=  4 * eye(len(self.States))
    
    def _CalculateObservationPosteriors(self, trainingFile):
        # Fetch and parse the training data into a set of means and covariances
        rawData = TrainingData.DataWrapper(trainingFile)
        classifications = rawData.GetFrequencies()
        classCount = len(classifications)
        
        means = [];
        covariances = [];
        for i in range(classCount):
            tuples = rawData.GetFrequency(classifications[i])
            means.append(mean(tuples))
            covariances.append(cov(tuples))
        
        # Use the means and covariances to fill in a probability density grid
        MAX_GRID_POINTS = 1000000
        gridResolution = int(MAX_GRID_POINTS ** (1.0 / classCount))
        
        # First calculate the numerical bounds of the grid
        #   Expand a [hyper]rectangul around each of the means 
        #     The size of the [hyper]rectangle is based on the main covariance terms
        #   Then wrap a bounding [hyper]square around all the [hyper]rectangles
        MAX_COVAR_STEPS = 3
        lower = float("inf")
        upper = float("-inf")
        for i in range(classCount):
            covariances[i]