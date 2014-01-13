function giveFeedback33(feedbackBoolean)

        global problem;
        global logger;

        if feedbackBoolean
            problem.feedback = problem.CORRECT_REWARD;
        else
            problem.feedback = problem.WRONG_PENALTY;
        end

        %%%% Step: update expected rewards given feedback %%%%
        %the structure of the update depends on whether the feedback was
        %positive or negative.  A negative feedback is simplest since it
        %affects only the 1 element of the reward matrix.
        if ~feedbackBoolean
            learningFactor = problem.LEARNING_RATE * problem.lastConfidence;
            problem.eRewards(problem.lastActionIndex,problem.lastClassIndex) = problem.eRewards(problem.lastActionIndex,problem.lastClassIndex) + learningFactor * (problem.feedback-problem.eRewards(problem.lastActionIndex,problem.lastClassIndex));
            
        %else we have to positively enforce the 1 element, and negatively
        %enforce all the other elements in the same row and column.
        else
            %positive reinforce to the one
            learningFactor = problem.LEARNING_RATE * problem.lastConfidence;
            problem.eRewards(problem.lastActionIndex,problem.lastClassIndex) = problem.eRewards(problem.lastActionIndex,problem.lastClassIndex) + learningFactor * (problem.feedback-problem.eRewards(problem.lastActionIndex,problem.lastClassIndex));
            
            %negative reinforce to the same column, row
            colMask = ones(size(problem.eRewards,1),1);
            rowMask = ones(1,size(problem.eRewards,2));
            
            colMask(problem.lastActionIndex) = 0;
            rowMask(problem.lastClassIndex) = 0;
            
            mask = zeros(size(problem.eRewards,1),size(problem.eRewards,2));
            mask(:,problem.lastClassIndex) = colMask;
            mask(problem.lastActionIndex,:) = rowMask;
            
            problem.eRewards = problem.eRewards + mask.*(learningFactor*(problem.WRONG_PENALTY*mask - problem.eRewards));
        end
        
        
        %%%% Step: set up POMDP problem given mapping selection %%%%
        rewards = [[problem.eRewards;repmat(problem.WAIT_PENALTY,1,length(problem.possibleStates) - length(problem.controls))],zeros(length(problem.actionLabels),length(problem.possibleStates) - length(problem.controls))];


        %%%% Step: calculate resulting policy for the POMDP problem %%%%
        %get latest policy
        if problem.isWindows
            [planes,actions] = generateModelDOS33(problem.p_obs_states,rewards);
        else
            [planes,actions] = generateModel33(problem.p_obs_states,rewards);
        end

        %%% Step: update logger %%%
        theTime = now;
        logger.expectedRewards{length(logger.expectedRewards)+1} = {theTime problem.eRewards};
        logger.feedback{length(logger.feedback)+1} = {theTime problem.feedback};
        logger.policyPlanes{length(logger.policyPlanes)+1} = {theTime planes};
        logger.policyActions{length(logger.policyActions)+1} = {theTime actions};
        
        drawDashboard33;