import json
from numpy import *
import numpy.testing

class DialogPOMDP:
    
    '''
    Loads a JSON file with the description of a POMDP dialog system
    Applies uncertainty over the brain state to observation probabilities
    
    Arguments:
        filename -> Assumes the following structure:
                        {
                            "Classifications" : [ <List of state labels for the POMDP> ], 
                            "Observations" : [ <List of labels> ], 
                            "Questions" : {
                                <Question label> : {
                                    "English" : <Description of the question>
                                    "Effect" : {
                                        <Element from "Classifications"> : {
                                            <Element from "Observations"> : <Probability of this element given the observation and question>
                                            <...> 
                                        }, 
                                        <...> 
                                    } 
                                }, 
                                <...>
                            }, 
                            "Rewards" : {
                                "Success"  : <Positive number>, 
                                "Failure"  : <Negative number>, 
                                "Question" : <Negative number greater than "Failure">, 
                                "Wait"     : <Tiny negative number>
                            }, 
                            "Time Discount" : <Number between 0 and 1 inclusive>
                        }
                    Note: Items surrounded by '<>' should be replaced with information about your model
                          All labels should only have alpha-numeric characters (and underscore)
        
        observeProb -> 2D square matrix
                       'observeProv[A, B]' is the conditional probability of observing A in state B
                       Rank of matrix should equal to the length of the "Observations" key in the JSON file (above)
                       Each column of the matrix should add to one
                           ie. The total probability of observing A in all states is 100%
    '''
    def __init__(self, filename, observeProb):
        filehandle = open(filename, "r")
        self._Model = json.load(filehandle)
        self._States = []
        self._Actions = []
        self._InitialBelief = [] 
        self._TransitionMatrix = {}  # [Action][Start, End]
        self._ObservationMatrix = [] # [State, Observation]
        self._RewardMatrix = []      # [Action, State]
        
        #######################################################################
        ##                      Top Level JSON Checking                      ##
        #######################################################################
        assert self._Model.has_key("Classifications")
        assert self._Model.has_key("Observations")
        assert self._Model.has_key("Questions")
        assert self._Model.has_key("Rewards")
        assert self._Model["Rewards"].has_key("Success")
        assert self._Model["Rewards"].has_key("Failure")
        assert self._Model["Rewards"].has_key("Question")
        assert self._Model["Rewards"]["Question"] > self._Model["Rewards"]["Failure"]
        assert self._Model["Rewards"].has_key("Wait")
        assert self._Model.has_key("Time Discount")
        
        # No whitespace is allowed in the labels
        for category in self._Model["Classifications"]:
            assert len(category.split()) == 1
        for observation in self._Model["Observations"]:
            assert len(observation.split()) == 1
        for question in self._Model["Questions"].keys():
            assert len(question.split()) == 1
        
        # Check the observation probabilities matrix
        assert len(observeProb.shape) == 2
        assert observeProb.shape[0] == observeProb.shape[1]
        assert len(self._Model["Observations"]) == observeProb.shape[0]
        numpy.testing.assert_almost_equal(sum(observeProb, axis=0), ones(observeProb.shape[0]))  
        
        # Check the time discount bounds
        assert self._Model["Time Discount"] >= 0 and self._Model["Time Discount"] <= 1
        
        #######################################################################
        ##            JSON Model Content Checking & Preprocessing            ##
        #######################################################################
        # Use the observation matrix "observeProb" to construct 
        #     a 2D observation probability matrix for each question
        #     To access the matrix -> self._Model["Questions"][Question]["Effect"]
        #     To determine the value of some cell -> matrix[Classification][Observation]
        # Also check the structure of the questions the POMDP can ask 
        for question in self._Model["Questions"].keys():
        
            # This does not have any effect on the POMDP, but should be included as a context item
            assert "English" in self._Model["Questions"][question]
            assert "Effect" in self._Model["Questions"][question]
            
            # Initialize the observation probability matrix such that each observation is equally likely
            questionObserveProb = ones((len(self._Model["Classifications"]), len(self._Model["Observations"])))
            questionObserveProb /= len(self._Model["Observations"])
            
            # For each specified classification, calculate an observation vector
            #    such that the n-th value represents P(noisy observation | classification)
            for category in self._Model["Questions"][question]["Effect"].keys():
            
                # Make sure all the states affected by the classification exist in the model
                assert category in self._Model["Classifications"]
                categoryInd = self._Model["Classifications"].index(category)
                
                # First fill in the observation vector with P(classification | noiseless observation)
                transProb = zeros(len(self._Model["Observations"]));
                for observation in self._Model["Questions"][question]["Effect"][category].keys():
                
                    # Make sure all the observations used by the question exist in the model
                    assert observation in self._Model["Observations"]
                    observeInd = self._Model["Observations"].index(observation)
                    
                    transProb[observeInd] = self._Model["Questions"][question]["Effect"][category][observation]
                    
                # Now multiply the observation vector by the observation matrix and normalize
                #   ie. Apply Bayes' rule
                # Place the observation vector into the observation probability matrix
                transProb = dot(transProb, observeProb)
                transProb /= sum(transProb)
                questionObserveProb[categoryInd, :] = transProb
            
            # Save the observation probability matrix in place of the data used to generate it
            self._Model["Questions"][question]["Effect"] = questionObserveProb
            
        #######################################################################
        ##                           POMDP States                            ##
        #######################################################################
        # General belief - while not asking a question, the general belief holds all the belief
        for state in self._Model["Classifications"]:
            self._States.append("S_" + state)
            
        # Question belief
        #     When the POMDP asks a question, the general belief transfers into a subset of the question belief
        #         A penalty is applied for transitioning into this state
        #     Observations change the belief state
        #     When the POMDP finishes with a question, the question belief transfers into the general belief
        for question in self._Questions:
            for state in self._Model["Classifications"]:
                self._States.append("q_" + question + "_S_" + state)
                
        # Terminal classification
        #     When the POMDP has asked enough questions to make a classification action
        #         the POMDP ends in one of these states
        for state in self._Model["Classifications"]:
            self._States.append("tS_" + state)
            
        #######################################################################
        ##                           POMDP Actions                           ##
        #######################################################################
        # Control ("wait" and "finish")
        #     When in the question belief states
        #         The POMDP may either "wait" for more observations
        #         Or "finish" asking to transfer back into the general belief states
        self._Actions.append("wait")
        self._Actions.append("finish")
        
        # Asking - when in the general belief states, the POMDP can ask any question
        for question in self._Model["Questions"].keys():
            self._Actions.append("qA_" + question)
            
        # Classifying - the POMDP may stop asking questions at any time and move to a terminal state
        for action in self._Model["Classifications"]:
            self._Actions.append("cA_" + action)
        
        #######################################################################
        ##                       POMDP Initial Belief                        ##
        #######################################################################
        # The model starts out uniform across the general belief and zero everywhere else
        startCount = double(len(self._Model["Classifications"]))
        for stateNum in range(len(self._States)):
            self._InitialBelief.append(str((stateNum < startCount) / startCount))
        
        #######################################################################
        ##            POMDP Action Transition Probability Matrix             ##
        #######################################################################
        # All actions result in deterministic transitions (probability 0 or 1)
        # The transition matrix is a map of 2D matrices
        #     Representation -> self._TransitionMatrix[Action][Start State][End State]
        GeneralBeliefEndIndex = len(self._Model["Classifications"])
        QuestionBeliefEndIndex = (1 + len(self._Questions)) * self._GeneralBeliefEndIndex
        
        # Control action "wait" - all states transition to themselves
        self._TransitionMatrix["wait"] = eye(len(self._States))
                
        # Control action "finish" - all question belief states transfer to the corresponding general belief state
        self._TransitionMatrix["finish"] = zeros((len(self._States), len(self._States)))
        self._TransitionMatrix["finish"][0:GeneralBeliefEndIndex, 0:GeneralBeliefEndIndex] = eye(GeneralBeliefEndIndex)
        self._TransitionMatrix["finish"][QuestionBeliefEndIndex:len(self._States), QuestionBeliefEndIndex:len(self._States)] = eye(len(self._States) - GeneralBeliefEndIndex)
        
        for questionIndex in range(len(self._Questions)):
            questionStateStartIndex = (1 + questionIndex) * len(self._Model["Classifications"])
            self._TransitionMatrix["finish"][questionStateStartIndex:(questionStateStartIndex + len(self._Model["Classifications"])), 0:GeneralBeliefEndIndex] = eye(GeneralBeliefEndIndex)
        
        # Asking a question will transfer all belief from general to question belief
        for questionIndex in range(len(self._Questions)):
            questionLabel = "qA_" + self._Questions[questionIndex]
            self._TransitionMatrix[questionLabel] = eye(len(self._States))
            self._TransitionMatrix[questionLabel][0:GeneralBeliefEndIndex, 0:GeneralBeliefEndIndex] = zeros((GeneralBeliefEndIndex, GeneralBeliefEndIndex))
            questionStateStartIndex = (1 + questionIndex) * len(self._Model["Classifications"])
            self._TransitionMatrix[questionLabel][0:GeneralBeliefEndIndex, questionStateStartIndex:(questionStateStartIndex + len(self._Model["Classifications"]))] = eye(GeneralBeliefEndIndex)
            
        # Classifying will transfer all general belief into a single terminal state
        for actionIndex in range(len(self._Model["Classifications"])):
            actionLabel = "cA_" + self._Model["Classifications"][actionIndex]
            self._TransitionMatrix[actionLabel] = eye(len(self._States))
            self._TransitionMatrix[actionLabel][0:GeneralBeliefEndIndex, 0:GeneralBeliefEndIndex] = zeros((GeneralBeliefEndIndex, GeneralBeliefEndIndex))
            self._TransitionMatrix[actionLabel][0:GeneralBeliefEndIndex, QuestionBeliefEndIndex + actionIndex] = ones(GeneralBeliefEndIndex)
                
        #######################################################################
        ##               POMDP Observation Probability Matrix                ##
        #######################################################################
        NumObservations = len(self._Model["Observations"])
        self._ObservationMatrix = zeros((len(self._States), NumObservations))
        
        # Observations mean nothing in general and terminal states
        self._ObservationMatrix[0:GeneralBeliefEndIndex, 0:NumObservations] = ones((GeneralBeliefEndIndex, NumObservations)) / float(NumObservations)
        self._ObservationMatrix[QuestionBeliefEndIndex:len(self._States), 0:NumObservations] = ones((GeneralBeliefEndIndex, NumObservations)) / float(NumObservations)
        
        for questionIndex in range(len(self._Questions)):
            question = self._Questions[questionIndex]
            categoryIndex = questionIndex * (1 + len(self._Model["Classifications"]))
            
            # Fill in the matrix with the piece-wise observation matrix
            self._ObservationMatrix[categoryIndex:(categoryIndex + len(self._Model["Classifications"])), NumObservations] = self._Model["Questions"][question]["Effect"]
        
        #######################################################################
        ##                        POMDP Reward Matrix                        ##
        #######################################################################
        # All non-relevant actions are penalized
        self._RewardMatrix = ones((len(self._Actions), len(self._States))) * self._Model["Rewards"]["Failure"]
        
        # Wait penalty
        self._RewardMatrix[self._Actions.index("wait"), 0:len(self._States)] = ones((1, len(self._States))) * self._Model["Rewards"]["Wait"]
        
        # Finish penalty
        self._RewardMatrix[self._Actions.index("finish"), 0:len(self._States)] = zeros((1, len(self._States)))
        
        # Asking penalty
        for question in self._Questions:
            self._RewardMatrix[self._Actions.index("qA_" + question), 0:len(self._States)] = ones((1, len(self._States))) * self._Model["Rewards"]["Question"]
        
        # Correct classification reward
        for action in self._Model["Classifications"]:
            actionIndex = self._Actions.index("cA_" + action)
            self._RewardMatrix[actionIndex, actionIndex] = self._Model["Rewards"]["Success"]
            
            # Reclassifying from terminal states is meaningless
            self._RewardMatrix[actionIndex, QuestionBeliefEndIndex:len(self._States)] = zeros((1, GeneralBeliefEndIndex))
        
    '''
    Writes a POMDP input file that can be solved with some POMDP solvers (i.e. APPL)
    '''
    def GenerateFile(self, filename):
        f = open(filename, "w")

        # Give a time discounting factor (to prevent the POMDP from gathering info infinitely)
        f.write("discount: " + str(self._Model["Time Discount"]) + "\n")
        
        # Use a reward function for maximization (as opposed to a cost function for minimization)
        f.write("values: reward\n")
        
        #################################################################
        ##                           Labels                            ##
        #################################################################
        f.write("states:")
        for state in self._States:
            f.write(" " + state)
        f.write("\n")
        
        f.write("actions:")
        for action in self._Actions:
            f.write(" " + action)
        f.write("\n")

        f.write("observations:")
        for observation in self._Model["Observations"]:
            f.write(" o_" + observation)
        f.write("\n")

        #################################################################
        ##                       Initial Belief                        ##
        #################################################################
        f.write("start:")
        for belief in self._InitialBelief:
            f.write(" " + belief)
        f.write("\n")
        
        #################################################################
        ##                      Transition matrix                      ##
        #################################################################
        for action in self._Actions:
            for priorStateIndex in range(len(self._States)):
                for postStateIndex in range(len(self._States)):
                    probability = self._TransitionMatrix[action][priorStateIndex, postStateIndex]
                    if (probability != 0): # Omit all the zeros
                        f.write("T : " + action + " : " + self._States[priorStateIndex] + " : " + self._States[postStateIndex] + " " + str(probability) + "\n")
        
        #################################################################
        ##               Observation Probability Matrix                ##
        #################################################################
        for stateIndex in range(len(self._States)):
            for observationIndex in range(len(self._Model["Observations"])):
                probability = self._ObservationMatrix[stateIndex, observationIndex]
                if (probability != 0): # Omit all the zeros
                    f.write("O : * : " + self._States[stateIndex] + " : o_" + self._Model["Observations"][observationIndex] + " " + str(probability)

        #################################################################
        ##                        Reward Matrix                        ##
        #################################################################
        for actionIndex in range(len(self._Actions)):
            for stateIndex in range(len(self._States)):
                reward = self._RewardMatrix[actionIndex, stateIndex]
                if (reward != 0): # Omit all the zeros
                    f.write("R : " self._Actions[actionIndex] + " : " + self._States[stateIndex] + " : * : * " + str(reward) + "\n")
