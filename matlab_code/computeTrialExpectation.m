
function [gamma,xi,prior,scale_a,score2,alpha2,scale3] = computeTrialExpectation(prior,likeli,transition)
% http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf
% dbstop if error

    t = 1;
    T = size(likeli,2);
    numstates = size(likeli,1);

    % E-step
    alpha = zeros(length(prior),T);
    alpha2 = zeros(length(prior),T);
    scale3 = zeros(length(prior),T);
    score2 = zeros(1,T);
    scale_a = ones(T,1);

    alpha(:,1) = prior .* likeli(:,1);
    alpha(:,t) = alpha(:,t) ./ sum(alpha(:,t));
    scale3(:,1) = alpha(:,1);

    alpha2(:,1) = prior;

    for t=2:T
        alpha(:,t) = transition(:,:,t)' * alpha(:,t-1);
        scale3(:,t) = alpha(:,t) ./ sum(alpha(:,t));
        alpha(:,t) = alpha(:,t) .* likeli(:,t);

        scale_a(t) = sum(alpha(:,t));
        alpha(:,t) = alpha(:,t) ./ scale_a(t);
        
        alpha2(:,t) = transition(:,:,t)' * alpha2(:,t-1);
        alpha2(:,t) = alpha2(:,t) ./ sum(alpha2(:,t));
        score2(t) = sum(alpha2(:,t) .* likeli(:,t));
    end

    beta = zeros(length(prior),T); % beta(i,t)  = Pr(O(t+1:T) | X(t)=i)
    beta(:,T) = ones(length(prior),1)/length(prior);

    scale_b = ones(T,1);
    for t=(T-1):-1:1
        beta(:,t) = transition(:,:,t+1) * (beta(:,t+1) .* likeli(:,t+1));
        scale_b(t) = sum(beta(:,t));
        beta(:,t) = beta(:,t) ./ scale_b(t);
    end

    alpha(alpha == 0) = eps(0);
    beta(beta == 0) = eps(0);
    gamma = exp(log(alpha) + log(beta) - repmat(log(cumsum((scale_a)))',[numstates,1]) - repmat(log(reverse(cumsum(reverse(scale_b))))',[numstates,1]));
    gamma(gamma == 0) = eps(0);
    gamma = gamma ./ repmat(sum(gamma,1),[numstates,1]);

    xi = zeros(length(prior),length(prior),T-1);
    transition2 = transition(:,:,2:end);

    for s1=1:numstates
        for s2=1:numstates
            xi(s1,s2,:) = log(likeli(s2,2:end)) + log(alpha(s1,1:end-1)) + log(squeeze(transition2(s1,s2,:))') + log(beta(s2,2:end)) - log(cumsum((scale_a(1:end-1))))' - log(reverse(cumsum(reverse(scale_b(2:end)))))';
            xi(s1,s2,:) = exp(xi(s1,s2,:));
        end
    end

    xi(xi == 0) = eps(0);
    xi = xi ./ repmat(sum(sum(xi,1),2),[numstates,numstates,1]);
    if size(xi,3) == 1
        xi = reshape(xi,[size(xi,1),size(xi,2),1]);
    end
    prior = gamma(:,1);
end