
%fftData is a vector in frequency domain where fftData{i+1} is the bucket for
% the i Hz frequency and is 1Hz wide (so fftData{1} is DC offset).
function action=doControl33(fftData)


    global problem;
    global logger;

    
    %%% Step: Update dashboard %%%
    %drawDashboard;
    
    
    %%%% Step: get P(Ot|Ct) using obs model %%%%
    %get obs vector
    obs = zeros(1,length(logger.frequenciesUsed));
    for i = 1:length(obs)
       obs(i) = fftData(logger.frequenciesUsed{i}+1);
    end
    
    %determine P(Ot|Ct) un-normalized
    p_obs = zeros(1,length(obs));
    for i = 1:length(p_obs)
        p_obs(i) = mvnpdf(obs,problem.means{i},problem.covs{i});
    end
    
    %normalize it
    normFactor = sum(p_obs);
    normMask = repmat(normFactor,1,length(p_obs));
    p_obs = p_obs ./ normMask;
       
    %%%% Step: update belief %%%%

    %set new belief
    newBelief = p_obs.*problem.belief;

    %normalize it
    problem.belief = newBelief/sum(newBelief);
    
    %display(problem.belief)


    %get latest policy
    planes = logger.policyPlanes{length(logger.policyPlanes)}{2};
    actions = logger.policyActions{length(logger.policyActions)}{2};
    
    
    %%%% Step: determine optimal action given policy and perform it %%%%
    %we augment belief with a zero for each state (these 0s represent the
    %  'C' - 'classified as' states - aka the states we are in after
    %  classifying.  We have 0 for these since we would not still be
    %  running if we classified).
    dots = [problem.belief,zeros(1,length(problem.possibleStates) - length(problem.belief))] * planes;
    q = find(dots==max(dots),1,'first');
    action = actions(q);
    theTime = now;

    logger.belief{length(logger.belief)+1} = {theTime problem.belief};
    
    if action==4
        problem.lastClass = 'Waiting';
        problem.controlMade = 'Wait';
        problem.lastActionIndex = action;
    else
        %softmax action selection (action = left, center, or right)
        
        %get weighted reward structure (essentially if we have high
        % confidence, this will be equal to the reward associated with each
        % action when in a given state.  But since we are not sure, we will
        % take a weighted average).
        Qs = problem.eRewards * problem.belief';
        
        %apply these quality scores to the Boltzmann distribution to get
        % the probability of selecting each one.
        Ws = exp(Qs/problem.BOLTZMANN_TEMPERATURE);
        Ws = Ws / sum(Ws);
        
        %now use these as probabilities to select an action.  Will probably
        % select the best action under most circumstances.
        selections = mnrnd(1,Ws);
        action = find(selections==1,1,'first');
        
        problem.lastActionIndex = action;
        problem.controlMade = problem.controls(problem.lastActionIndex);
        problem.lastClassIndex = find(problem.belief==max(problem.belief),1,'first');
        problem.lastClass = problem.lastClassIndex;
        
%         display(problem.belief)
%         for i = 0:0.05:1
%             doink = [i 1-i 0 0] * planes;
%             q2 = find(doink==max(doink),1,'first');
%             fprintf('%f %d\n',i,actions(q2));
%         end    
        problem.lastConfidence = max(problem.belief);
        problem.belief = repmat(1/length(logger.frequenciesUsed),1,length(logger.frequenciesUsed));
    end
    
    %%% Step: log some info %%%
    problem.observation = p_obs;
    
    logger.observation{length(logger.observation)+1} = {theTime fftData};
    logger.confidence{length(logger.confidence)+1} = {theTime p_obs};
    logger.actionMade{length(logger.actionMade)+1} = {theTime problem.lastActionIndex};
    logger.policyPlanes{length(logger.policyPlanes)+1} = {theTime planes};
    logger.policyActions{length(logger.policyActions)+1} = {theTime actions};
    
    drawDashboard33;
        
    