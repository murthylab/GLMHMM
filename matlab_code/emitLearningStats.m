function [allvalue, allhess] = emitLearningStats(emit_w, stim, statenum, options)
% trans_w are the weights that we are learning: in format states x weights
% stim is a cell with each stimulus (stim{}.data) and the probability
%       transition functions (stim{}.gamma and stim{}.xi)

% note that this transition function is dependent on where we are
% transitioning FROM, and relies on each of the other possible states we
% could be transitioning to; so we cannot minimize these independently.
% Thus we are really going to find the gradient of all transition filters
% originating FROM some state
%     stim_e = repmat(reshape(stim{trial},1,1,size(stim{trial},1),size(stim{trial},2)),[numstates,numsymb-1,1,1]);

% http://www.ism.ac.jp/editsec/aism/pdf/044_1_0197.pdf

    numstates = stim{1}.numstates;
%     numbins = stim{1}.numtotalbins;
    numbins = size(stim{1}.data,1);
    % states x bins
    emit_w = reshape(emit_w,numbins,numstates)';

    allhess = zeros(numstates,numstates,numbins);
    allvalue = 0;
    totalT = 0;

    lambda = options.emit_lambda;

    for trial=1:length(stim)
        totalT = totalT + size(stim{trial}.data,2);
    end

    for trial=1:length(stim)
%         tgrad = zeros(numstates,numbins);
        T = size(stim{trial}.data,2);
        % convert into states x bins x time and sum across bins
        filtpower = reshape(sum(reshape(repmat(emit_w,[1,1,T]),[numstates,numbins,T]) .* repmat(reshape(stim{trial}.data(:,:),1,numbins,T),[numstates,1,1]),2),numstates,T);
        % HACK: change this...
        filtpower(filtpower > 600) = 600;
        % now filtpower is states x time

%         value = sum(stim{trial}.gamma(statenum,:) .* -log(squeeze(1 + sum(exp(filtpower),1))));
        value = stim{trial}.gamma(statenum,:) .* -log(squeeze(1 + sum(exp(filtpower),1)));
%         value = value + sum(stim{trial}.gamma(statenum,stim{trial}.emit == 0));
        norm = repmat(1 + sum(exp(filtpower),1),[numstates,1]);
        hess_t = (exp(filtpower).^2 - exp(filtpower).*norm)./norm.^2;
        hess_t(isnan(hess_t)) = 0;
        hess = zeros(numstates,numstates,numbins);

        for i=1:size(filtpower,1)
            hess(i,i,:) = sum(repmat(stim{trial}.gamma(statenum,:),[numbins,1]).* repmat(hess_t(i,:),[size(stim{trial}.data,1),1]) .* stim{trial}.data.^2,2);
            for j=1:size(filtpower,1)
                if j == i
                    continue
                end
                hess(i,j,:) = sum(repmat(stim{trial}.gamma(statenum,:),[numbins,1]).* repmat(exp(filtpower(i,:)).*exp(filtpower(j,:))./norm(i,:).^2,[size(stim{trial}.data,1),1]) .* stim{trial}.data.^2,2);
            end

            value(stim{trial}.emit == i) = value(stim{trial}.emit == i) + stim{trial}.gamma(statenum,stim{trial}.emit == i) .* filtpower(i,stim{trial}.emit == i);
        end
        value = sum(value);

        allhess = allhess + hess;
        allvalue = allvalue + value;
    end

%     allgrad = -allgrad/totalT;
%     allgrad(:,1:end-1) = allgrad(:,1:end-1)  + lambda * emit_w(:,1:end-1);
%     allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,1:end-1).^2));

    if options.autoanneal
        % need to remember to remove this...
%         if (lambda > 0.5)
%             allgrad(:,:) = allgrad(:,:)  + lambda * emit_w(:,:);
%             allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,:).^2));
%         elseif (lambda > 0.1)
%             sensorynumber = 510;
%             allgrad(:,[1:sensorynumber end]) = allgrad(:,[1:sensorynumber end])  + lambda * emit_w(:,[1:sensorynumber end]);
%             allgrad(:,[(sensorynumber+1):(end-1)]) = allgrad(:,[(sensorynumber+1):(end-1)])  + 2*lambda * emit_w(:,[(sensorynumber+1):(end-1)]);
%             allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,[1:sensorynumber end]).^2));
%             allvalue = allvalue + 2*(lambda/2) * sum(sum(emit_w(:,(sensorynumber+1):(end-1)).^2));
%         else
%             sensorynumber = 510;
%             allgrad(:,[1:sensorynumber end]) = allgrad(:,[1:sensorynumber end])  + lambda * emit_w(:,[1:sensorynumber end]);
%             allgrad(:,[(sensorynumber+1):(end-1)]) = allgrad(:,[(sensorynumber+1):(end-1)])  + 5*lambda * emit_w(:,[(sensorynumber+1):(end-1)]);
%             allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,[1:sensorynumber end]).^2));
%             allvalue = allvalue + 5*(lambda/2) * sum(sum(emit_w(:,(sensorynumber+1):(end-1)).^2));
%         end
        allhess(:,:,:) = -allhess(:,:,:)/totalT^2 + lambda;
        allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,:).^2));
    else
        allhess(:,:,:) = -allhess(:,:,:)  + lambda;
        allvalue = -allvalue/totalT + (lambda/2) * sum(sum(emit_w(:,:).^2));
    end

    for i=1:size(allhess,3)
        allhess(:,:,i) = squeeze(allhess(:,:,i))^-1;
    end
    if any(isnan(allhess(:))) || any(isnan(allvalue))
        display('WTF! SOMETHING BAD HAPPENED! ISNAN! EEEE')
    end

    if nargout > 1
        for i=1:size(allhess,1)
            outhess(i,:) = allhess(i,i,:);
        end
        allhess = reshape(outhess',size(outhess,1)*size(outhess,2),1);
    end
end