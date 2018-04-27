function [transition] = GLMHMM_TransLik(trans_w,stim)
    T = size(stim,2);
    numstates = size(trans_w,1);
    numbins = size(trans_w,3);
    transition = zeros(numstates,numstates,T);
    for i=1:numstates
        filtpower = squeeze(sum(repmat(reshape(trans_w(i,:,:),numstates,numbins),[1,1,T]) .* repmat(reshape(stim,1,size(stim,1),T),[numstates,1,1]),2));
        if numstates == 1
            filtpower = filtpower';
        end

        filtpower(i,:) = 0;
        for j=1:numstates
            transition(i,j,:) = exp(filtpower(j,:) - logsumexp(filtpower,1));
        end
    end
end