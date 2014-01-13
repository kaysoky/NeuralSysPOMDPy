% Load APPL Policy
%   Matt Bryan - December, 2011
%
% The POMDP policy estimator called "APPL" outputs its policies in an xml
%   format.  This MATLAB function reads an APPL policy file and returns a matrix 'planes'
%   and a vector 'actions.'  APPL represents policies using a piecewise
%   linear convex function.  The function maps belief states (that is - a
%   probability distribution over all states) to expected total future
%   reward given optimal actions are taken, starting at that belief state. 
%   Each hyperplane making up this piecewise function is defined by an
%   "alpha vector" with number of components equal to the number of
%   possible states (in other words - the dimensionality of the belief
%   state space). The returned 'planes' matrix contains one column for each
%   plane.  That column contains the alpha vector defining that plane.  The
%   rows represent each state in the same order they were given in the
%   input .pomdp file.  The 'actions' vector contains the action that
%   corresponds to each hyperplane.
%
%   You can use this policy to take optimal actions. Optimality here means the action which
%   maximizes total expected future reward.  To use this policy, take the dot 
%   product of your belief state against each of these alpha vectors.  The highest 
%   such dot product corresponds to the most rewarding plane on that spot in belief 
%   space.  Its corresponding index in the action vector is the optimal
%   action to take given your belief state.  For example, if you compute
%   the dot product of your belief state against each alpha vector and find
%   planes(56) gives you the highest result, then actions(56) is the
%   optimal action to take given your current belief state.
%
%   To better understand the interpretation of these planes, go here and
%   note what each line segment represents and what is the optimal action
%   at each spot in the belief state space:
%   http://www.pomdp.org/pomdp/tutorial/pomdp-vi-example.shtml
%   The algorithm I give in the comments above is the process of finding
%   the "highest" line on such graphs, only we are working in potentially
%   higher dimensional spaces.

function [planes,actions]=loadAPPLPolicy(fileName)
    f = fopen(fileName,'r');
    
    planes = [];   %states x planes
    done = false;
    actions = [];
    numberOfPlanes = 0;
    numberOfStates = 0;
    i = 0;  

    while(~done)
        currentLine = fgets(f);
        if currentLine == -1
            break;
        end
        
        toks = strread(currentLine,'%s');

        if strcmp(toks(1),'<AlphaVector')==1
            numberOfStates = str2double(strrep(strrep(toks(2), 'vectorLength="', ''),'"',''));
            numberOfPlanes = str2double(strrep(strrep(toks(4), 'numVectors="', ''),'">',''));
            planes = zeros(numberOfStates,numberOfPlanes);
            actions = zeros(numberOfPlanes,1);
            continue;
        end


        if strcmp(toks(1),'<Vector')==1
            i = i+1;
            actionLabel = str2double(strrep(strrep(toks(2), 'action="', ''),'"',''));
            actions(i) = actionLabel+1;

            for j = 3:(numberOfStates+2)

               if(j==3)
                   expectedReward = str2double(strrep(strrep(toks(j), 'obsValue="0">', ''),'"',''));
                   planes(j-2,i) = expectedReward;
               else
                   expectedReward = str2double(toks(j));
                   planes(j-2,i) = expectedReward;
               end

            end
        end
    end 
end