function initControlModel33(trainingData,isWindows)

    global problem;
    global logger;
    global currentTask;
    global lastTask;
    global dashboard;
    
    dashboard = figure('Name','Model Dashboard','NumberTitle','off');
    
    display('Setting up model')
    
    problem = struct('belief',[],'lastClass',[],'lastActionIndex',[], ...
        'eRewards',[],'WAIT_PENALTY',[],'CORRECT_REWARD',[],...
        'WRONG_PENALTY',[],'LEARNING_RATE',[],'TIME_DISCOUNT'...
        ,[],'controls',[],'feedback',[],'possibleStates',[],...
        'possibleObservations',[],'observation',[],'actionLabels',[],...
        'p_obs_states',[],'mappingIndexMap',[],'controlsIndexMap',[],...
        'mappingLabels',[],'controlMade',[],'selectedMapping',[],...
        'policyPlanes',[],'policyActions',[],'confidence',[],'lastConfidence',...
        [],'BOLTZMANN_TEMPERATURE',[]);

    
    %which windows in each training trial to use for training
    TRAIN_WINDOWS = 1:23;
    
    %model constants
    problem.WAIT_PENALTY = -0.000001;
    problem.CORRECT_REWARD = 10;
    problem.WRONG_PENALTY = -30;
    problem.LEARNING_RATE = 0.2;
    problem.TIME_DISCOUNT = 0.99;
    problem.STRATA_PER_CLASS = 7;
    problem.OBS_CARDINALITY = 15;
    problem.BOLTZMANN_TEMPERATURE = 1.3;
    
   
    %toggle windows vs. unix mode for calling POMDP solver
    problem.isWindows = isWindows;


    %RL aspects for learning reward function
    %'eRewards' gives us the expected reward from making a given control while
    %  in a given state.  control x state
    %we initialize to just above wait penalty so we are incentivizing initial
    %  exploration
    problem.eRewards = repmat(-3,3,3);
    problem.eRewards(1,1) = 1;
    problem.eRewards(2,2) = 1;
    problem.eRewards(3,3) = 1;

    %2 control features
    problem.controls = {'left','center','right'};

    %initialize for display
    currentTask = '.';
    lastTask = '.';


    %pomdp model specs
    %'belief' - prob of being in brain states; uniform prior
    problem.belief = repmat(1/size(problem.eRewards,2),1,size(problem.eRewards,2));
    
    %% now parse the training data to develop the observation model

    display('Training Gaussian obs model')
    
    %chunk up data into each class
    groups = cell(length(trainingData),1);        %each group of data in raw form
    conditions = cell(length(trainingData),1);    %each group label (frequency)
    masterData = cell(length(trainingData),1);    %each group of data subsetted out to relevant frequencies
    problem.means = cell(length(trainingData),1);
    problem.covs = cell(length(trainingData),1);
    
    for i = 1:length(trainingData)
       groups{i} = trainingData(i).data;
       conditions{i} = str2num(strtok(trainingData(i).condition,' '));
    end
    
    classCount = length(conditions);
    
    %reshape data into usable form (list of tuples in feature space), and
    %  calculate means and covariances for each class
    for i = 1:classCount
       masterData{i} = [];
       for j = 1:classCount
          masterData{i} = [masterData{i}; groups{i}(conditions{j}+1,:,:)];
       end
       
       
       masterData{i} = reshape(permute(masterData{i}(:,TRAIN_WINDOWS,:),[3,2,1]),size(masterData{i},3)*numel(TRAIN_WINDOWS),size(masterData{i},1));
       
       problem.means{i} = mean(masterData{i});
       problem.covs{i} = cov(masterData{i});
    end
    
   
    display('Generating grid')

    %calculate grid of values for discretization process / numeric integration
    TOTAL_POINT_COUNT = 1000000;
    gridResolution = ceil(TOTAL_POINT_COUNT ^ (1/classCount));

    covMasks = cell(classCount,1);
    ranges = cell(classCount,1);
    ZVals = cell(classCount,1);
    lower = 100000000;
    upper = -100000000;
    
    % calculate grid bounds
    for i = 1:classCount
       covMasks{i} = diag(problem.covs{i})' * 3;
       covMasks{i} = [covMasks{i}*-1;covMasks{i}];
       
       ranges{i} = covMasks{i} + repmat(problem.means{i},2,1);
       
       lower = min(min(min(ranges{i})),lower);
       upper = max(max(max(ranges{i})),upper);
    end
    
    % generate grid
    gridVals = lower:(upper-lower)/gridResolution:upper;
    
    gridResolution = length(gridVals);
    actualPointCount = gridResolution ^ classCount;

    %you don't want to know what this does (N-dimensional meshgrid)
    coords = zeros(actualPointCount,classCount);
    gridIters = gridResolution .^ fliplr(0:classCount-1);
    for i = 1:classCount
        currIndex = 1;
        for q = 1:gridIters(classCount-i+1);
            for j = 1:gridResolution
                for k = 1:gridIters(i)
                    coords(currIndex,i) = gridVals(j);
                    currIndex = currIndex+1;
                end
            end
        end
    end

    display('Building posteriors at grid points')
    for i = 1:classCount
        ZVals{i} = mvnpdf(coords,problem.means{i},problem.covs{i});
    end
    
    % calculate the posteriors
    pointCount = size(coords,1);
    posteriors = zeros(pointCount,classCount);
    
    for i = 1:classCount
        posteriors(:,i) = ZVals{i};
    end

    norm = sum(posteriors,2);
    normMask = repmat(norm,1,classCount);
    posteriors = posteriors./normMask;

    for i = 1:pointCount
        for j = 1:classCount
           posteriors(i,j) = round(((problem.STRATA_PER_CLASS-1) *  posteriors(i,j))+1); 
        end
    end
    
    usedLabels = unique(posteriors,'rows');
    
    
    display('Clustering discrete chunks to have fewer of them')
    
    ZLabelsOld = zeros(size(posteriors,1),1);
    for i = 1:size(posteriors,1)
           currentZ = posteriors(i,:);
           [~,currentLabel]=ismember(currentZ,usedLabels,'rows');
           ZLabelsOld(i) = currentLabel;
    end
    
    discMeans = grpstats(coords,ZLabelsOld);
    newLabels = kmeans(discMeans,problem.OBS_CARDINALITY);

    ZLabels = zeros(size(ZLabelsOld));
    for i = 1:length(ZLabels);
        ZLabels(i) = newLabels(ZLabelsOld(i));
    end
	obsStateSize = problem.OBS_CARDINALITY;
    
    display('Numerically integrating')
    
    % numerically integrate posterior surface within each discrete region,
    %  which is a stratum of the posterior.  We integrate over each class's
    %  PDF.  This gives us the observation model mapping P(o|c). 
    strataIntegrals = zeros(obsStateSize,classCount);
    
    
    %this is the footprint of each column making up the Riemann sum.
    sampleVolume = (gridVals(2)-gridVals(1))^(classCount); 

    % get the value of each element of the sums
    %  For a 2d riemann sum, this is equal to the length of the base of
    %  each rectangle times its height - the area of each rectangle
    samples = cell(classCount,1);
    for i = 1:classCount
       samples{i} = ZVals{i} * sampleVolume; 
    end
    

    %this gives us the integral approximation for the whole PDF.
    %sum(strataIntegrals(:,1)), for instance, should be roughly 1.
    % now integrate within each discrete region
    for i = 1:classCount
        for j = 1:length(samples{i})
           currentLabel = ZLabels(j);
           strataIntegrals(currentLabel,i) = strataIntegrals(currentLabel,i) + samples{i}(j);
        end
    end
    
    display('Writing out discretized obs model and other parameters')
    
    % normalize out the observation model since we don't integrate out to
    % infinity, only out to +/- 3 STDEVs in each direction.  Algebra is too
    % hard for a real integral.
    normFactor = sum(strataIntegrals);
    normMask = repmat(normFactor,obsStateSize,1);
    obsProbs = strataIntegrals ./normMask;   %this is the final result that gets used in the POMDP
    
    %% finish writing parameters

    problem.actionLabels = {'a_l';'a_c';'a_r';'a_w'};
    problem.possibleStates = {'s_12','s_17','s_20','s_l','s_c','s_r'};
    
    problem.possibleObservations = cell(obsStateSize,1);  
    for i = 1:obsStateSize
        problem.possibleObservations{i} = num2str(i);
    end
    

    %set dashboard-related constants, start up figure for plotting
    problem.lastClass = '.';
    problem.lastActionIndex = length(problem.actionLabels);


    %convenience vectors for later
    problem.mappingIndexMap = 1:length(problem.eRewards);
    problem.controlsIndexMap = 1:length(problem.controls);

    %last feedback recieved.
    problem.feedback = 0;

    %probability that simple max likelihood classification would make a mistake
    %dim1 = obs  [1,2,3,...]; dim2=possible states
    
    problem.p_obs_states = [obsProbs repmat(1/size(obsProbs,1),size(obsProbs,1),length(problem.controls))];
    
    %set up the logger
    logger.observation = cell(1);
    logger.expectedRewards = cell(1);
    logger.feedback = cell(1);
    logger.belief = cell(1);
    logger.actionMade = cell(1);
    logger.policyPlanes = cell(1);
    logger.policyActions = cell(1);
    logger.confidence = cell(1);
    
    %output the initialized records
    theTime = now;
    logger.observation{1} = {theTime problem.observation};
    logger.expectedRewards{1} = {theTime problem.eRewards'};
    logger.feedback{1} = {theTime problem.feedback};
    logger.belief{1} = {theTime problem.belief};
    logger.actionMade{1} = {theTime problem.lastActionIndex};
    logger.frequenciesUsed = conditions;
    
        
    %initialize the POMDP system
    %rewards =
    %[problem.CORRECT_REWARD,problem.WRONG_PENALTY,0,0;problem.WRONG_PENALTY,problem.CORRECT_REWARD,0,0;problem.WAIT_PENALTY,problem.WAIT_PENALTY,0,0];
    %rewards = [0,0,0,0;0,0,0,0;problem.WAIT_PENALTY,problem.WAIT_PENALTY,0,0];
    rewards = [[problem.eRewards;repmat(problem.WAIT_PENALTY,1,length(problem.possibleStates) - length(problem.controls))],zeros(length(problem.actionLabels),length(problem.possibleStates) - length(problem.controls))];
    
    
    %%%% Step: calculate resulting policy for the POMDP problem %%%%
    %get latest policy
    if problem.isWindows
        [planes,actions] = generateModelDOS33(problem.p_obs_states,rewards);
    else
        [planes,actions] = generateModel33(problem.p_obs_states,rewards);
    end
    
    logger.policyPlanes{length(logger.policyPlanes)+1} = {theTime planes};
    logger.policyActions{length(logger.policyActions)+1} = {theTime actions};
    
    display('Control model initialization complete -----------')