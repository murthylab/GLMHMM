
function [allvalue, allgrad] = transLearningFun(trans_w, stim, statenum, options)
% trans_w are the weights that we are learning: in format states x weights
% stim is a cell with each stimulus (stim{}.data) and the probability
%       transition functions (stim{}.gamma and stim{}.xi)

% note that this transition function is dependent on where we are
% transitioning FROM, and relies on each of the other possible states we
% could be transitioning to; so we cannot minimize these independently.
% Thus we are really going to find the gradient of all transition filters
% originating FROM some state

    numstates = stim{1}.numstates;
    numbins = stim{1}.numtotalbins;

    trans_w = reshape(trans_w,numbins,numstates)';

    allgrad = zeros(numstates, numbins);
    allvalue = 0;
    totalT = 0;

    lambda = options.trans_lambda;

    for trial=1:length(stim)
        totalT = totalT + size(stim{trial}.data,2);
    end

    for trial=1:length(stim)
        tgrad = zeros(length(trans_w),1);
        T = size(stim{trial}.data,2)-1;
        % use data from 1:end-1 or 2:end?
%         filtpower = squeeze(sum(repmat(trans_w,[1,1,T]) .* repmat(reshape(stim{trial}.data(:,1:end-1),1,numbins,T),[numstates,1,1]),2));
        filtpower = squeeze(sum(repmat(trans_w,[1,1,T]) .* repmat(reshape(stim{trial}.data(:,2:end),1,numbins,T),[numstates,1,1]),2));
        % now filtpower is states x time

        
%         squeeze(stim{trial}.xi(2,1,:))' .* log(exp(filtpower(1,:))./squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1))) + ...
%             squeeze(stim{trial}.xi(2,2,:))' .* log(1./squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1)))
% 
%         squeeze(stim{trial}.xi(2,1,:))' .* log(exp(filtpower(1,:))) - ...
%         squeeze(stim{trial}.xi(2,1,:))' .* log(squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1))) - ...
%             squeeze(stim{trial}.xi(2,2,:))' .* log(squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1)))
% 
%         squeeze(stim{trial}.xi(2,1,:))' .* log(exp(filtpower(1,:))) - ...
%         stim{trial}.gamma(i,1:end-1) .* log(squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1)))

%         value = -stim{trial}.gamma(statenum,2:end) .* log(1 + squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),2))');
        value = -stim{trial}.gamma(statenum,1:end-1) .* log(squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1)));
%         value = -squeeze(stim{trial}.xi(statenum,statenum,:))' .* log(squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1)));
        if size(stim{trial}.xi,3) == 1
            tgrad = squeeze(stim{trial}.xi(statenum,:,:))';
        else
            tgrad = squeeze(stim{trial}.xi(statenum,:,:));
        end
        % should it be gamma(2:end) or gamma(1:end-1)?
        i=statenum;
%         offset = stim{trial}.gamma(i,2:end) ./ squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,i),:)),1));
        offset = stim{trial}.gamma(i,1:end-1) ./ squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,i),:)),1));
%         offset = squeeze(stim{trial}.xi(i,i,:))' ./ squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,i),:)),1));
        for j=1:numstates
            if (i~=j)
%                 value = value + squeeze(stim{trial}.xi(statenum,j,:))' .*log(exp(filtpower(j,:)));
                value = value + squeeze(stim{trial}.xi(statenum,j,:))' .*filtpower(j,:);
                tgrad(j,:) = tgrad(j,:) - exp(filtpower(j,:)).*offset;
            else
                tgrad(j,:) = 0;
            end
        end
        tgrad = sum(repmat(reshape(tgrad,numstates,1,T),[1,numbins,1]) .* repmat(reshape(stim{trial}.data(:,2:end),1,numbins,T),[numstates,1,1]),3);

        % I probably don't need to rescale here because that happens
        % naturally but... oh well.
        allgrad = allgrad + tgrad;
        allvalue = allvalue + sum(value);
    end

%     allgrad = -allgrad/totalT;
%     allgrad(:,1:end-1) = allgrad(:,1:end-1) + lambda * trans_w(:,1:end-1);
%     allgrad(statenum,:) = 0;
%     allvalue = -allvalue/totalT + (lambda/2) * sum(sum(trans_w(:,1:end-1).^2));

%     allgrad = -allgrad/totalT;
%     allgrad(:,:) = allgrad(:,:) + lambda * trans_w(:,:);
%     allgrad(statenum,:) = 0;
%     allvalue = -allvalue/totalT + (lambda/2) * sum(sum(trans_w(:,:).^2));
    gradRegularization = zeros(size(allgrad));
    valueRegularization = 0;

    if options.L2smooth
        Dx1 = spdiags(ones(size(trans_w,2)-1,1)*[-1 1],0:1,size(trans_w,2)-1-1,size(trans_w,2)-1); 
        Dx = Dx1'*Dx1; % computes squared diffs
        
        for fstart=(options.numFilterBins+1):options.numFilterBins:(size(trans_w,2)-1)
            Dx(fstart,fstart) = 1;
            Dx(fstart-1,fstart-1) = 1;
            Dx(fstart-1,fstart) = 0;
            Dx(fstart,fstart-1) = 0;
        end
        D = blkdiag(Dx,0);

        if options.AR_lambda ~= -1
            if length(options.smoothLambda) == 1
                options.smoothLambda = repmat(options.smoothLambda(1),[size(trans_w,1)-1,size(trans_w,2)]);
                options.smoothLambda(:,options.ARvec) = options.AR_lambda;
            end

            gradRegularization(setdiff(1:numstates,statenum),:) = gradRegularization(setdiff(1:numstates,statenum),:) + options.smoothLambda .* (D * trans_w(setdiff(1:numstates,statenum),:)')';
            valueRegularization = valueRegularization + sum(sum(((options.smoothLambda/2) .* (D * trans_w(setdiff(1:numstates,statenum),:)')').^2));
        else
            gradRegularization(setdiff(1:numstates,statenum),:) = gradRegularization(setdiff(1:numstates,statenum),:) + options.smoothLambda .* (D * trans_w(setdiff(1:numstates,statenum),:)')';
            valueRegularization = valueRegularization +  sum(sum(((options.smoothLambda/2) .* (D * trans_w(:,:)')').^2));
        end
    end
    if lambda ~= 0
        if options.AR_lambda ~= -1
            gradRegularization = gradRegularization + [lambda * trans_w(:,options.stimvec), options.AR_lambda * trans_w(:,options.ARvec)];
            valueRegularization = valueRegularization + (lambda/2) * sum(sum(trans_w(:,options.stimvec).^2)) + (options.AR_lambda/2) * sum(sum(trans_w(:,options.ARvec).^2));
        else
            gradRegularization = gradRegularization + lambda * trans_w(:,:);
            valueRegularization = valueRegularization + (lambda/2) * sum(sum(trans_w(:,:).^2));
        end
    end

    allgrad = -allgrad/totalT + gradRegularization;
    allvalue = -allvalue/totalT + valueRegularization;

    if allvalue < 0
        display('why oh why oh why');
    end
%     allgrad = -allgrad/totalT;
%     allvalue = -allvalue/totalT;

    if nargout > 1
        allgrad = reshape(allgrad',size(allgrad,1)*size(allgrad,2),1);
    end
end