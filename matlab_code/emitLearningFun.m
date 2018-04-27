function [allvalue, allgrad] = emitLearningFun(emit_w, stim, statenum, options)
% trans_w are the weights that we are learning: in format states x weights
% stim is a cell with each stimulus (stim{}.data) and the probability
%       transition functions (stim{}.gamma and stim{}.xi)

% note that this transition function is dependent on where we are
% transitioning FROM, and relies on each of the other possible states we
% could be transitioning to; so we cannot minimize these independently.
% Thus we are really going to find the gradient of all transition filters
% originating FROM some state
%     stim_e = repmat(reshape(stim{trial},1,1,size(stim{trial},1),size(stim{trial},2)),[numstates,numsymb-1,1,1]);

%     display('entering emit');
    numstates = stim{1}.numstates;
    numbins = stim{1}.numtotalbins;
    % states x bins
    emit_w = reshape(emit_w,numbins,numstates)';

    allgrad = zeros(numstates, numbins);
    allvalue = 0;
    totalT = 0;

    lambda = options.emit_lambda;

    % find out how many data points we are dealing with so that we can normalize
    % (I don't think we actually need to do this but it helps keep regularization values consistent from fit to fit)
    for trial=1:length(stim)
        totalT = totalT + size(stim{trial}.data,2);
    end

    for trial=1:length(stim)
        T = size(stim{trial}.data,2);
        % convert into states x bins x time and sum across bins
        filtpower = reshape(sum(reshape(repmat(emit_w,[1,1,T]),[numstates,numbins,T]) .* repmat(reshape(stim{trial}.data(:,:),1,numbins,T),[numstates,1,1]),2),numstates,T);
        % now filtpower is states x time

        %
        value = stim{trial}.gamma(statenum,:) .* -log(squeeze(1 + sum(exp(filtpower),1)));
        tgrad = -exp(filtpower) ./ repmat(1 + sum(exp(filtpower),1),[numstates,1]);

%         value = value + sum(stim{trial}.gamma(statenum,stim{trial}.emit == 0));
        for i=1:size(filtpower,1)
            tgrad(i,stim{trial}.emit == i) = 1 + tgrad(i,stim{trial}.emit == i);
%             value = value + sum(stim{trial}.gamma(statenum,stim{trial}.emit == i) .* sum(log(exp(filtpower(i,stim{trial}.emit == i))),1));
            value(stim{trial}.emit == i) = value(stim{trial}.emit == i) + stim{trial}.gamma(statenum,stim{trial}.emit == i) .* filtpower(i,stim{trial}.emit == i);
        end
        value = sum(value);
        if any(isnan(value(:)))
            display('ugh');
        end
        tgrad = tgrad.* repmat(stim{trial}.gamma(statenum,:),[numstates,1]);

        tgrad = sum(repmat(reshape(tgrad,numstates,1,T),[1,numbins,1]) .* repmat(reshape(stim{trial}.data,1,numbins,T),[numstates,1,1]),3);

        allgrad = allgrad + tgrad;
        allvalue = allvalue + value;
    end
    
%     display('done computing');

    gradRegularization = 0;
    valueRegularization = 0;
    if options.L2smooth

        Dx1 = spdiags(ones(size(emit_w,2)-1,1)*[-1 1],0:1,size(emit_w,2)-1-1,size(emit_w,2)-1); 
        Dx = Dx1'*Dx1; % computes squared diffs

        for fstart=(options.numFilterBins+1):options.numFilterBins:(size(emit_w,2)-1)
            Dx(fstart,fstart) = 1;
            Dx(fstart-1,fstart-1) = 1;
            Dx(fstart-1,fstart) = 0;
            Dx(fstart,fstart-1) = 0;
        end
        D = blkdiag(Dx,0);
        
%             allgrad = allgrad + options.smoothLambda .* (D * emit_w')';
%             allvalue = allvalue + (options.smoothLambda/2) * sum(sum((D * emit_w(:,:)').^2));

        if options.AR_lambda ~= -1
            if length(options.smoothLambda) == 1
                options.smoothLambda = repmat(options.smoothLambda(1),[size(emit_w,1),size(emit_w,2)]);
                options.smoothLambda(:,options.ARvec) = options.AR_lambda;
            end
            
            gradRegularization = gradRegularization + options.smoothLambda .* (D * emit_w')';
            valueRegularization = valueRegularization + sum(sum(((options.smoothLambda/2) .* (D * emit_w(:,:)')').^2));
        else
            gradRegularization = gradRegularization + options.smoothLambda .* (D * emit_w')';
            valueRegularization = valueRegularization +  sum(sum(((options.smoothLambda/2) .* (D * emit_w(:,:)')').^2));
        end
    end
    if lambda ~= 0
        if options.AR_lambda ~= -1
            gradRegularization = gradRegularization + [lambda * emit_w(:,options.stimvec), options.AR_lambda * emit_w(:,options.ARvec)];
            valueRegularization = valueRegularization + (lambda/2) * sum(sum(emit_w(:,options.stimvec).^2)) + (options.AR_lambda/2) * sum(sum(emit_w(:,options.ARvec).^2));
        else
            gradRegularization = gradRegularization + lambda * emit_w(:,:);
            valueRegularization = valueRegularization + (lambda/2) * sum(sum(emit_w(:,:).^2));
        end
    end

    allgrad = -allgrad/totalT + gradRegularization;
    allvalue = -allvalue/totalT + valueRegularization;

    if any(isnan(allgrad(:))) || any(isnan(allvalue))
        display('WTF! SOMETHING BAD HAPPENED! ISNAN! OH')
    end

    if nargout > 1
        allgrad = reshape(allgrad',size(allgrad,1)*size(allgrad,2),1);
    end
end