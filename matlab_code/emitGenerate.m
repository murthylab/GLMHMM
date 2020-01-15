function [output,state] = emitGenerate(emit_w, trans_w, data, symb, useAR)
% trans_w are the weights that we are learning: in format states x weights
% stim is a cell with each stimulus (stim{}.data) and the probability
%       transition functions (stim{}.gamma and stim{}.xi)

% note that this transition function is dependent on where we are
% transitioning FROM, and relies on each of the other possible states we
% could be transitioning to; so we cannot minimize these independently.
% Thus we are really going to find the gradient of all transition filters
% originating FROM some state
%     stim_e = repmat(reshape(stim{trial},1,1,size(stim{trial},1),size(stim{trial},2)),[numstates,numsymb-1,1,1]);

    featurebins = 120;

    if ~exist('useAR','var')
        useAR = false;
    end

% we are going to have to handcode some features here for a bit...
    if useAR
        p1_mean = mean(symb == 1);
        p2_mean = mean(symb == 3);
        s_mean = mean(symb == 2);
        a_mean = mean(symb ~= 0);

        p1_std = std(symb == 1);
        p2_std = std(symb == 3);
        s_std = std(symb == 2);
        a_std = std(symb ~= 0);
        
        if p2_mean == 0
            s_start = size(emit_w,3)-1 - featurebins*3;
            p1_start = size(emit_w,3)-1 - featurebins*2;
            a_start = size(emit_w,3)-1 - featurebins;

            s_score = unique(data(s_start-1 + (1:featurebins),:));
            p1_score = unique(data(p1_start-1 + (1:featurebins),:));
            a_score = unique(data(a_start-1 + (1:featurebins),:));
        else
            s_start = size(emit_w,3)-1 - featurebins*4;
            p1_start = size(emit_w,3)-1 - featurebins*3;
            p2_start = size(emit_w,3)-1 - featurebins*2;
            a_start = size(emit_w,3)-1 - featurebins;

            s_score = unique(data(s_start-1 + (1:featurebins),:));
            p1_score = unique(data(p1_start-1 + (1:featurebins),:));
            p2_score = unique(data(p2_start-1 + (1:featurebins),:));
            a_score = unique(data(a_start-1 + (1:featurebins),:));
        end
    end

    numstates = size(trans_w,1);
    numemissions = size(emit_w,2);
    numbins = size(trans_w,3);
    T = size(data,2);

    output = zeros(T,1);
    state = zeros(T,1);

    psample = rand(T,1);
    psample_state = rand(T,1);
    for s1=1:numstates
        sguess(s1) = 1./(1+sum(exp(sum(reshape(trans_w(s1,setdiff(1:numstates,s1),:),[numstates-1,numbins]).*repmat(data(:,1)',[numstates-1,1]),2)),1));
    end
    tmp = find(sguess == max(sguess));
    state(1) = tmp(1);      % whatever, I guess this is just random at this point...
    for t=1:T-featurebins
        filtpower = squeeze(exp(sum(emit_w(state(t),:,:).*repmat(reshape(data(:,t)',[1,1,size(data,1)]),[1,numemissions,1]),3)))';
        lik2 = [0;filtpower ./ squeeze(1 + sum(filtpower))];
%         outsymb = find(lik2 == max(lik2));
        outsymb = find(cumsum(lik2) < psample(t));
        outsymb = outsymb(end);
        if outsymb == length(lik2)
            outsymb = 0;
        end
        output(t) = outsymb;

%         lik3 = [1;filtpower] ./ squeeze(1 + sum(filtpower));
%         [~,outsymb] = max(lik3);
%         output(t) = outsymb-1;
        
        if useAR
            for b=1:featurebins
    %             if outsymb == 1
    %                 data(p1_start + featurebins-b,t+b) = p1_score(2);
    %             else
    %                 data(p1_start + featurebins-b,t+b) = p1_score(1);
    %             end
    %             if outsymb == 2
    %                 data(s_start + featurebins-b,t+b) = s_score(2);
    %             else
    %                 data(s_start + featurebins-b,t+b) = s_score(1);
    %             end
    %             if outsymb == 3
    %                 data(p2_start + featurebins-b,t+b) = p2_score(2);
    %             else
    %                 data(p2_start + featurebins-b,t+b) = p2_score(1);
    %             end
    %             if outsymb == 0
    %                 data(a_start + featurebins-b,t+b) = a_score(1);
    %             else
    %                 data(a_start + featurebins-b,t+b) = a_score(2);
    %             end
                if output(t) == 1
                    data(p1_start + b - 1,t+b) = p1_score(2);
                else
                    data(p1_start + b - 1,t+b) = p1_score(1);
                end
                if output(t) == 2
                    data(s_start + b - 1,t+b) = s_score(2);
                else
                    data(s_start + b - 1,t+b) = s_score(1);
                end
                if p2_mean ~= 0
                    if output(t) == 3
                        data(p2_start + b - 1,t+b) = p2_score(2);
                    else
                        data(p2_start + b - 1,t+b) = p2_score(1);
                    end
                end
                if output(t) == 0
                    data(a_start + b - 1,t+b) = a_score(1);
                else
                    data(a_start + b - 1,t+b) = a_score(2);
                end
            end
        end

        filtpower = exp(sum(reshape(trans_w(state(t),setdiff(1:numstates,state(t)),:),[numstates-1,numbins]).*repmat(data(:,t)',[numstates-1,1]),2));
        ind = 0;
        for s1=setdiff(1:numstates,state(t))
            ind = ind + 1;
            sguess(s1) = filtpower(ind)./(1+sum(filtpower,1));
        end
        sguess(state(t)) = 1./(1+sum(filtpower,1));
        tmp = find(cumsum([0,sguess]) < psample_state(t));
        state(t+1) = tmp(end);
    end
end