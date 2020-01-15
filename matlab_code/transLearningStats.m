function [allvalue, allhess] = transLearningStats(trans_w, stim, statenum, options)
% trans_w are the weights that we are learning: in format states x weights
% stim is a cell with each stimulus (stim{}.data) and the probability
%       transition functions (stim{}.gamma and stim{}.xi)

% note that this transition function is dependent on where we are
% transitioning FROM, and relies on each of the other possible states we
% could be transitioning to; so we cannot minimize these independently.
% Thus we are really going to find the gradient of all transition filters
% originating FROM some state

    numstates = stim{1}.numstates;
%     numbins = stim{1}.numtotalbins;
    numbins = size(stim{1}.data,1);

    trans_w = reshape(trans_w,numbins,numstates)';

%     allgrad = zeros(numstates, numbins);
    allhess = zeros(numstates-1,numstates-1,numbins);
    allvalue = 0;
    totalT = 0;

    lambda = options.trans_lambda;

    for trial=1:length(stim)
        totalT = totalT + size(stim{trial}.data,2);
    end

    setorder = setdiff(1:numstates,statenum);
    for trial=1:length(stim)
        T = size(stim{trial}.data,2)-1;
        % use data from 1:end-1 or 2:end?
%         filtpower = squeeze(sum(repmat(trans_w,[1,1,T]) .* repmat(reshape(stim{trial}.data(:,1:end-1),1,numbins,T),[numstates,1,1]),2));
        filtpower = squeeze(sum(repmat(trans_w,[1,1,T]) .* repmat(reshape(stim{trial}.data(:,2:end),1,numbins,T),[numstates,1,1]),2));
        % now filtpower is states x time

%         value = -stim{trial}.gamma(statenum,2:end) .* log(1 + squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),2))');
        value = -stim{trial}.gamma(statenum,1:end-1) .* log(squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1)));
%         value = -squeeze(stim{trial}.xi(statenum,statenum,:))' .* log(squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1)));
        norm = squeeze(1 + sum(exp(filtpower(setdiff(1:numstates,statenum),:)),1));

        hess = zeros(numstates-1,numstates-1,numbins);
        % we have to 'reindex' because we now have a states-1 x states-1
        % matrix, where the diagonal is whichever state filter we are currently
        % interested in...
        datvec = reshape(stim{trial}.data(:,2:end),1,numbins,T).^2;
        for i=1:length(setorder)
            for j=1:length(setorder)
                if (i~=j)
    %                 value = value + squeeze(stim{trial}.xi(statenum,j,:))' .*log(exp(filtpower(j,:)));
                    hess(i,j,:) = sum(repmat(reshape(stim{trial}.gamma(statenum,1:end-1,:) .* exp(filtpower(setorder(i),:)) .* exp(filtpower(setorder(j),:)) ./ norm.^2,[1,1,T]),[1,numbins,1]) .* datvec,3);
                else
                    hess(i,i,:) = sum(repmat(reshape(stim{trial}.gamma(statenum,1:end-1,:) .* (norm.*exp(filtpower(setorder(i),:)) - exp(filtpower(setorder(i),:)).^2) ./ norm.^2,[1,1,T]),[1,numbins,1]) .* datvec,3);
                end
            end
        end

        % I probably don't need to rescale here because that happens
        % naturally but... oh well.
        allhess = allhess + hess;
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

    if options.autoanneal
%         if (lambda > 0.5)
%             allgrad(:,:) = allgrad(:,:)  + lambda * trans_w(:,:);
%             allvalue = -allvalue/totalT + (lambda/2) * sum(sum(trans_w(:,:).^2));
%         elseif (lambda > 0.1)
%             sensorynumber = 510;
%             allgrad(:,[1:sensorynumber end]) = allgrad(:,[1:sensorynumber end])  + lambda * trans_w(:,[1:sensorynumber end]);
%             allgrad(:,[(sensorynumber+1):(end-1)]) = allgrad(:,[(sensorynumber+1):(end-1)])  + 2*lambda * trans_w(:,[(sensorynumber+1):(end-1)]);
%             allvalue = -allvalue/totalT + (lambda/2) * sum(sum(trans_w(:,[1:sensorynumber end]).^2));
%             allvalue = allvalue + 2*(lambda/2) * sum(sum(trans_w(:,(sensorynumber+1):(end-1)).^2));
%         else
%             sensorynumber = 510;
%             allgrad(:,[1:sensorynumber end]) = allgrad(:,[1:sensorynumber end])  + lambda * trans_w(:,[1:sensorynumber end]);
%             allgrad(:,[(sensorynumber+1):(end-1)]) = allgrad(:,[(sensorynumber+1):(end-1)])  + 5*lambda * trans_w(:,[(sensorynumber+1):(end-1)]);
%             allvalue = -allvalue/totalT + (lambda/2) * sum(sum(trans_w(:,[1:sensorynumber end]).^2));
%             allvalue = allvalue + 5*(lambda/2) * sum(sum(trans_w(:,(sensorynumber+1):(end-1)).^2));
%         end
        allhess = -allhess + lambda;
        allvalue = -allvalue/totalT + (lambda/2) * sum(sum(trans_w(:,:).^2));
    else
        allhess = -allhess + lambda;
        allvalue = -allvalue/totalT + (lambda/2) * sum(sum(trans_w(:,:).^2));
    end

    for i=1:size(allhess,3)
        allhess(:,:,i) = squeeze(allhess(:,:,i))^-1;
    end

    if allvalue < 0
        display('why oh why oh why');
    end

    if nargout > 1
        for i=1:length(setorder)
            outhess(setorder(i),:) = allhess(i,i,:);
        end
        outhess(statenum,:) = 0;
        allhess = reshape(outhess',size(outhess,1)*size(outhess,2),1);
    end
end