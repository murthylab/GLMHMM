function [allvalue, allgrad] = emitMultistepLearningFun(emit_w, stim, statenum, options)
% trans_w are the weights that we are learning: in format states x weights
% stim is a cell with each stimulus (stim{}.data) and the probability
%       transition functions (stim{}.gamma and stim{}.xi)

% note that this transition function is dependent on where we are
% transitioning FROM, and relies on each of the other possible states we
% could be transitioning to; so we cannot minimize these independently.
% Thus we are really going to find the gradient of all transition filters
% originating FROM some state
%     stim_e = repmat(reshape(stim{trial},1,1,size(stim{trial},1),size(stim{trial},2)),[numstates,numsymb-1,1,1]);

    % I will have to do something to make this more generic to work
    % with other formats
    numsteps = options.numsteps;
    numsamples = options.numsamples;

    numstates = stim{1}.numstates;
    numbins = stim{1}.numtotalbins;
    % states x bins
    emit_w = reshape(emit_w,numbins,numstates)';

    allgrad = zeros(numstates, numbins);
    allvalue = 0;
    totalT = 0;

    lambda = options.emit_lambda;

    for trial=1:length(stim)
        totalT = totalT + size(stim{trial}.data,2);
    end

    for trial=1:length(stim)
        % basically, for each step we are looking ahead, we are going to
        % generate a sample and then use that to calculate the lookahead
        % likelihood
        % since we are using large amounts of data here, we can get away
        % with only using one sample (I think)
        
        % I might have to use ADAM for SGD?
        %  https://www.mathworks.com/matlabcentral/fileexchange/61616-adam-stochastic-gradient-descent-optimization
        for sample=1:numsamples
            newstim = stim{trial}.data;
            newemit = [];
%             for step=1:numsteps
                
                
                
                % two steps:
                % first, find the likelihood of the actual data at STEPs 
                % away
                % second, find the likelihood of all generated data...

                % FIRST
%                 if step == 1
                    T = size(newstim,2);
                    % convert into states x bins x time and sum across bins
                    filtpower = reshape(sum(reshape(repmat(emit_w,[1,1,T]),[numstates,numbins,T]) .* repmat(reshape(newstim,1,numbins,T),[numstates,1,1]),2),numstates,T);
                    % now filtpower is states x time

                    value = stim{trial}.gamma(statenum,:) .* -log(squeeze(1 + sum(exp(filtpower),1)));
                    tgrad = -exp(filtpower) ./ repmat(1 + sum(exp(filtpower),1),[numstates,1]);

                    for i=1:size(filtpower,1)
                        tgrad(i,stim{trial}.emit == i) = 1 + tgrad(i,stim{trial}.emit == i);
                        value(stim{trial}.emit == i) = value(stim{trial}.emit == i) + stim{trial}.gamma(statenum,stim{trial}.emit == i) .* filtpower(i,stim{trial}.emit == i);
                    end
                    value = sum(value);
                    tgrad = tgrad.* repmat(stim{trial}.gamma(statenum,:),[numstates,1]);

                    tgrad = sum(repmat(reshape(tgrad,numstates,1,T),[1,numbins,1]) .* repmat(reshape(newstim,1,numbins,T),[numstates,1,1]),3);

                    allgrad = allgrad + tgrad;
                    allvalue = allvalue + value;
%                 else
%                     [newvalue,newgrad] = generatePosteriorNStep_backup(stim{trial}.data,stim{trial}.emit,step-1,emit_w,stim{trial}.gamma(statenum,:));
                    [newvalue,newgrad] = generatePosteriorNStep(stim{trial}.data,stim{trial}.emit,numsteps-1,emit_w,stim{trial}.gamma(statenum,:));
                    for cnum=1:(numsteps-1)
                        allgrad = allgrad + newgrad{cnum};
                        allvalue = allvalue + sum(newvalue{cnum});
                    end
%                 end
% 
%                 % SECOND
%                 if step > 1
%                     T = size(oldstim,2);
%                     % convert into states x bins x time and sum across bins
%                     filtpower = reshape(sum(reshape(repmat(emit_w,[1,1,T]),[numstates,numbins,T]) .* repmat(reshape(oldstim,1,numbins,T),[numstates,1,1]),2),numstates,T);
%                     % now filtpower is states x time
%  
%                     value = oldgamma(statenum,:) .* -log(squeeze(1 + sum(exp(filtpower),1)));
%                     tgrad = -exp(filtpower) ./ repmat(1 + sum(exp(filtpower),1),[numstates,1]);
% 
%                     for i=1:size(filtpower,1)
%                         tgrad(i,newemit == i) = 1 + tgrad(i,newemit == i);
%                         value(newemit == i) = value(newemit == i) + oldgamma(statenum,newemit == i) .* filtpower(i,newemit == i);
%                     end
%                     value = sum(value);
%                     tgrad = tgrad.* repmat(oldgamma(statenum,:),[numstates,1]);
% 
%                     tgrad = sum(repmat(reshape(tgrad,numstates,1,T),[1,numbins,1]) .* repmat(reshape(oldstim,1,numbins,T),[numstates,1,1]),3);
% 
%                     allgrad = allgrad + tgrad * (numsteps - step + 1);
%                     allvalue = allvalue + value * (numsteps - step + 1);
%                 end
% 
%                 if (step ~= numsteps) && (numsteps > 1)
%                     oldstim = newstim;
%                     oldgamma = stim{trial}.gamma;
% 
%                     [newstim,newemit] = generateNextStep(newstim,emit_w,numstates,numbins);
% 
%                     newemit = newemit - 1;
% 
%                     newstim = newstim(:,2:end);
%                     stim{trial}.gamma = stim{trial}.gamma(:,2:end);
%                     stim{trial}.emit = stim{trial}.emit(2:end);
%                 end
%             end
        end
    end

    % implement smoothing: block matrix that is
    % lambda_2 * [[1,-1,...],[-1,2,-1,...],[0,-1,2,-1,0,...]]
    % I need to make sure that this matrix takes into account the boundary
    % size...
    if options.autoanneal
        allgrad(:,:) = -allgrad(:,:)/totalT + lambda * emit_w(:,:);
        allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,:).^2));
    elseif options.L2smooth
        allgrad(:,:) = -allgrad(:,:)/totalT + lambda * emit_w(:,:);
        allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,:).^2));

        Dx1 = spdiags(ones(size(emit_w,2)-1,1)*[-1 1],0:1,size(emit_w,2)-1-1,size(emit_w,2)-1); 
        Dx = Dx1'*Dx1; % computes squared diffs
        
        for fstart=(options.numFilterBins+1):options.numFilterBins:(size(emit_w,2)-1)
            Dx(fstart,fstart) = 1;
            Dx(fstart-1,fstart-1) = 1;
            Dx(fstart-1,fstart) = 0;
            Dx(fstart,fstart-1) = 0;
        end
        D = blkdiag(Dx,0);

        allgrad = allgrad + options.smoothLambda .* (D * emit_w')';
        allvalue = allvalue + (options.smoothLambda/2) * sum(sum((D * emit_w(:,:)').^2));
    else
        allgrad(:,:) = -allgrad(:,:)/totalT  + lambda * emit_w(:,:);
        allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,:).^2));
    end

    if any(isnan(allgrad(:))) || any(isnan(allvalue))
        display('WTF! SOMETHING BAD HAPPENED! ISNAN! OH')
    end

    if nargout > 1
        allgrad = reshape(allgrad',size(allgrad,1)*size(allgrad,2),1);
    end
end