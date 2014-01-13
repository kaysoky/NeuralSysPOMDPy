function [planes,actions] = generateModelDOS33(p_obs_states, rewards)

global problem;

brainStateCount = 3;
terminalStateCount = 3;
actionCount = length(problem.actionLabels);

stateCount = brainStateCount + terminalStateCount;

initialBelief = [repmat(1/brainStateCount,1,brainStateCount) repmat(0.0,1,terminalStateCount)];

%deterministic transitions (always stay in same state, except during
%  classification where we transition to a terminal state)
p_transitions = zeros(length(problem.actionLabels),stateCount,stateCount);

%set self-loops for terminal states for all actions
p_transitions(:,brainStateCount+1:stateCount,brainStateCount+1:stateCount) = repmat(permute(eye(terminalStateCount),[3,1,2]),[actionCount,1,1]);

%set self-loops for brain states for the wait action (last action)
p_transitions(actionCount,1:brainStateCount,1:brainStateCount) = permute(eye(brainStateCount),[3,1,2]);

%set transitions to terminal states when a classification action is made
p_transitions(1:actionCount-1,1:brainStateCount,brainStateCount+1:stateCount) = repmat(permute(eye(terminalStateCount),[1,3,2]),[1,brainStateCount,1]);



generatePOMDPInputs('test.pomdp', problem.TIME_DISCOUNT, problem.possibleStates, problem.actionLabels, problem.possibleObservations, p_transitions, p_obs_states, rewards,initialBelief);

dos('pomdpsol test.pomdp --output out.policy --timeout 0.08 > $null');

[planes,actions]=loadAPPLPolicy('out.policy');