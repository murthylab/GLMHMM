function [allvalue, alllik, alllik2, alllik3, alllik4] = emitLikelihood(emit_w, stim, statenum)
% trans_w are the weights that we are learning: in format states x weights
% stim is a cell with each stimulus (stim{}.data) and the probability
%       transition functions (stim{}.gamma and stim{}.xi)

% allvalue = <log likelihood>
% alllik = <pure likelihood>
% alllik2 = <MAP decision f'n of likelihood>
% alllik3 = <likelihood | song type>
% alllik4 = <MAP decision f'n | song type>

    numstates = stim{1}.numstates;
%     numbins = stim{1}.numtotalbins;
    numbins = size(stim{1}.data,1);
    % states x bins
    emit_w = reshape(emit_w,numbins,numstates)';

    neighborhood = 1;

    allvalue = 0;
    alllik = 0;
    alllik2 = 0;
    alllik3 = zeros(numstates+1,numstates+1);
    alllik4 = zeros(numstates+1,numstates+1);
    totalT = 0;
    totalT3 = zeros(numstates+1,1);

    for trial=1:length(stim)
        totalT = totalT + size(stim{trial}.data,2);
        totalT3(end) = totalT3(end) + sum(stim{trial}.emit == 0);
        for i=1:numstates
            totalT3(i) = totalT3(i) + sum(stim{trial}.emit == i);
        end
    end

    for trial=1:length(stim)
%         tgrad = zeros(numstates,numbins);
        T = size(stim{trial}.data,2);
        % convert into states x bins x time and sum across bins
        filtpower = reshape(sum(reshape(repmat(emit_w,[1,1,T]),[numstates,numbins,T]) .* repmat(reshape(stim{trial}.data(:,:),1,numbins,T),[numstates,1,1]),2),numstates,T);
        % now filtpower is states x time

        value = stim{trial}.gamma(statenum,:) .* -log(squeeze(1 + sum(exp(filtpower),1)));
        lik = 1 ./ squeeze(1 + sum(exp(filtpower),1));
        lik2 = repmat(reshape(1 ./ squeeze(1 + sum(exp(filtpower),1)),[1,T]),[numstates+1,1]);
        lik3 = zeros(size(filtpower,1)+1,size(filtpower,1)+1);
        lik3(end,end) = sum(lik(stim{trial}.emit == 0).* stim{trial}.gamma(statenum,stim{trial}.emit == 0));
        for j=1:size(filtpower)
            lik3(end,j) = sum(lik(stim{trial}.emit == 0).* exp(filtpower(j,stim{trial}.emit == 0)).* stim{trial}.gamma(statenum,stim{trial}.emit == 0));
            lik3(j,end) = sum(lik(stim{trial}.emit == j).* stim{trial}.gamma(statenum,stim{trial}.emit == j));
        end

        lik4 = zeros(size(filtpower,1)+1,size(filtpower,1)+1);

        for i=1:size(filtpower,1)
            value(stim{trial}.emit == i) = value(stim{trial}.emit == i) + stim{trial}.gamma(statenum,stim{trial}.emit == i) .* log(exp(filtpower(i,stim{trial}.emit == i)));
            for j=1:size(filtpower,1)
                lik3(i,j) = sum(lik(stim{trial}.emit == i).* exp(filtpower(j,stim{trial}.emit == i)).* stim{trial}.gamma(statenum,stim{trial}.emit == i));
            end
            lik(stim{trial}.emit == i) = lik(stim{trial}.emit == i) .* exp(filtpower(i,stim{trial}.emit == i));
            lik2(i,:) = lik2(i,:) .* exp(filtpower(i,:));
        end
        for i=1:size(filtpower,1)
            lik4(i,end) = sum((lik2(end,stim{trial}.emit == i) == max(lik2(:,stim{trial}.emit == i),[],1)).* stim{trial}.gamma(statenum,stim{trial}.emit == i));
            lik4(end,i) = sum((lik2(i,stim{trial}.emit == 0) == max(lik2(:,stim{trial}.emit == 0),[],1)).* stim{trial}.gamma(statenum,stim{trial}.emit == 0));
            for j=1:size(filtpower,1)
                lik4(i,j) = sum((lik2(j,stim{trial}.emit == i) == max(lik2(:,stim{trial}.emit == i),[],1)).* stim{trial}.gamma(statenum,stim{trial}.emit == i));
            end
        end
        lik4(end,end) = sum((lik2(end,stim{trial}.emit == 0) == max(lik2(:,stim{trial}.emit == 0),[],1)).* stim{trial}.gamma(statenum,stim{trial}.emit == 0));
        lik2 = (lik == max(lik2,[],1)).* stim{trial}.gamma(statenum,:);
        lik = lik.* stim{trial}.gamma(statenum,:);

        value = sum(value);
        lik = sum(lik);
        lik2 = sum(lik2);

        % I probably don't need to rescale here because that happens
        % naturally but... oh well.
        allvalue = allvalue + value;
        alllik = alllik + lik;
        alllik2 = alllik2 + lik2;
        alllik3 = alllik3 + lik3;
        alllik4 = alllik4 + lik4;
    end

    allvalue = -allvalue/totalT;
    alllik = alllik/totalT;
    alllik2 = alllik2/totalT;
    alllik3 = alllik3./repmat(totalT3,[1,size(alllik3,2)]);
    alllik4 = alllik4./repmat(totalT3,[1,size(alllik4,2)]);

    if any(isnan(alllik(:))) || any(isnan(allvalue))
        display('WTF! SOMETHING BAD HAPPENED! ISNAN! AH')
    end
end