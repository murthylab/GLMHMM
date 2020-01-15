function [newvalue,newgrad] = generatePosteriorNStep(stim,emit,steps,emit_w,gamma)
% given that we want to generate STEPS worth of latent data and we know
% what is happening at STEPS+1, how do we sample from the posterior?
% p(y_l | Y_t+m, Y_t) ~ p(Y_t+m | y_l, Y_t) * p(y_l | Y_t)
% so for each possible y_l find the probability of generating that sample
% and the probability of seeing Y_t+m given that sample
% then draw from (the normalized version of) this probability distribution

% what if instead of going through all possible latents we just sample from
% them? And hope that on average we converge on the right distribution...
% Then instead of needing 3^N possibilities for N steps away we can just
% get away with some smaller number, eg 1+N or even just two options...
%
% (If this works, I could use the same technique for the analog signal)

    numpaths = 3;

    symb = unique(emit);

    % start with step 1 being of length T
    % then generate a stimulus and make the next set of values T-1
    % we will use this to predict T+1, meaning we only need to use
    % emit(2:end)

    currvec = [];
    value = 0;
    grad = [];
    for ll=1:steps
        if ll == 1
            % at the first time step,
            % we want to take the generate likelihoods, gradients, and
            % values and remove one time bin
            
            % we only need to generate 1:end-1, because generating a
            % synthetic value for the last bin will NEVER be used (because
            % there is no emission in the following bin that will use that
            % latent prediction)
            oldstim = stim;

            currvec = zeros(numpaths,ll,size(oldstim,2)-1);
            p_each = zeros(size(currvec,1),size(oldstim,2)-1);
            autostim = zeros(size(currvec,1),size(oldstim,1),size(oldstim,2)-1);
            grad = zeros(size(currvec,1),size(emit_w,1),size(emit_w,2),size(oldstim,2)-1);
            value = zeros(numpaths,size(oldstim,2)-1);

            for vv=1:numpaths
                choice = floor(rand(1,size(oldstim,2)-1)*length(symb));

                pnew = stimlikelihood(oldstim(:,1:end-1),emit_w);

                % not sure either this or the value are quite right...
                grad(vv,:,:,:) = [reshape(repmat(((choice == 1)-pnew(2,:).*gamma(1:(end-ll))),[size(emit_w,2),1]).*oldstim(:,1:(end-1)),[1,size(emit_w,2),size(oldstim,2)-1]); ...
                                     reshape(repmat(((choice == 2)-pnew(3,:)).*gamma(1:(end-ll)),[size(emit_w,2),1]).*oldstim(:,1:(end-1)),[1,size(emit_w,2),size(oldstim,2)-1])];

                % I have to do that stupid indexing thing here...
                plik = pnew;
                value(vv,choice(1:length(plik)) == 0) = gamma(choice(1:length(plik)) == 0).*log(plik(1,choice(1:length(plik)) == 0));
                value(vv,choice(1:length(plik)) == 1) = gamma(choice(1:length(plik)) == 1).*log(plik(2,choice(1:length(plik)) == 1));
                value(vv,choice(1:length(plik)) == 2) = gamma(choice(1:length(plik)) == 2).*log(plik(3,choice(1:length(plik)) == 2));

                p_each(vv,:) = plik(sub2ind(size(plik),choice+1,1:length(plik)));
                currvec(vv,1,:) = choice;
                
                % now generate what the new stimulus would look like given
                % this sequence of choices
                autostim(vv,:,:) = makeNewStimFromVec(oldstim(:,1:end-1),choice);
            end
            
            % at the end of this, we want to save time by generating a
            % sequence using the posterior
            [newemit,reallik,holdlik] = chooseNewStim(autostim,emit_w,p_each,emit);
            [newgrad{ll}, newvalue{ll}] = updateGrad(holdlik, reallik, newemit, autostim, grad, value, gamma, emit);
        else
            % at every time step (after the first)
            % we want to take the previous likelihoods, gradients, and
            % values and remove one time bin
            
            % at this time step, we no longer care about whatever is in the
            % first bin: if we are generating ll+1 stimuli at this point in
            % the algorithm, we will have to rely on ll bins! So the first
            % ll bins WON'T have latent stimuli but rather REAL stimuli

            oldstim = autostim(:,:,2:end);
%             for vv=1:numpaths
%                 oldstim(vv,:,:) = shiftAutoStim(stim,squeeze(autostim(vv,:,:)),ll);
%             end
            oldgrad = grad(:,:,:,1:end-1);
            oldvalue = value(:,1:end-1);
            p_each_old = p_each(:,1:end-1);

            p_each = zeros(size(currvec,1),size(oldstim,3));
            autostim = zeros(size(currvec,1),size(oldstim,2),size(oldstim,3));
            grad = zeros(size(currvec,1),size(emit_w,1),size(emit_w,2),size(oldstim,3));
            value = zeros(numpaths,size(oldstim,2));
            currvec = currvec(:,:,1:end-1);

            for vv=1:numpaths
                % now we have taken the stimuli from the previous time step
                % and shifted them into place as if they were already
                % generated and are in the autocorr stimuli parameters:
                choice = floor(rand(1,size(oldstim,3))*length(symb));
                plik = stimlikelihood(squeeze(oldstim(vv,:,:)),emit_w);

                value(vv,choice(1:length(plik)) == 0) = gamma(choice(1:length(plik)) == 0).*log(plik(1,choice(1:length(plik)) == 0)) + oldvalue(vv,choice(1:length(plik)) == 0);
                value(vv,choice(1:length(plik)) == 1) = gamma(choice(1:length(plik)) == 1).*log(plik(2,choice(1:length(plik)) == 1)) + oldvalue(vv,choice(1:length(plik)) == 1);
                value(vv,choice(1:length(plik)) == 2) = gamma(choice(1:length(plik)) == 2).*log(plik(3,choice(1:length(plik)) == 2)) + oldvalue(vv,choice(1:length(plik)) == 2);

                p_each(vv,:) =  plik(sub2ind(size(plik),choice+1,1:length(plik))).* squeeze(p_each_old(vv,:));

                grad(vv,:,:,:) = squeeze(oldgrad(vv,:,:,:)) + [reshape(repmat(((choice == 1)-plik(2,:).*gamma(ll:(end-1))),[size(emit_w,2),1]).*squeeze(oldstim(vv,:,:)),[1,size(emit_w,2),size(stim,2)-ll]); ...
                                     reshape(repmat((choice==2)-plik(3,:).*gamma(ll:(end-1)),[size(emit_w,2),1]).*squeeze(oldstim(vv,:,:)),[1,size(emit_w,2),size(stim,2)-ll])];

                currvec(vv,ll,:) = choice;
                autostim(vv,:,:) = makeNewStimFromVec(squeeze(oldstim(vv,:,:)),choice);
            end
            
            % at the end of this, we want to save time by generating a
            % sequence using the posterior
            [newemit,reallik,holdlik] = chooseNewStim(autostim,emit_w,p_each,emit);
            [newgrad{ll}, newvalue{ll}] = updateGrad(holdlik, reallik, newemit, autostim, grad, value, gamma, emit);
        end
    end
end

function [newstim] = makeNewStimFromVec(oldstim,symb)
if size(oldstim,1) > 602
    symb0 = 601:630;
    symb1 = 541:600;
    symb2 = 511:540;
else
    symb0 = 571:600;
    symb1 = 541:570;
    symb2 = 511:540;
end

    newstim = oldstim;
    newstim([symb0(1);symb1(1);symb2(1)] + 1,2:end) = oldstim([symb0(1);symb1(1);symb2(1)],1:end-1);

    newstim(symb0(1),symb~=0) = max(max(newstim(symb0,:)));
    newstim(symb0(1),symb==0) = min(min(newstim(symb0,:)));

    newstim(symb1(1),symb==1) = max(max(newstim(symb1,:)));
    newstim(symb1(1),symb~=1) = min(min(newstim(symb1,:)));

    newstim(symb2(1),symb==2) = max(max(newstim(symb2,:)));
    newstim(symb2(1),symb~=2) = min(min(newstim(symb2,:)));
end

% function [newstim] = shiftAutoStim(stim,oldstim,step)
% if size(oldstim,1) > 602
%     symb0 = 601:630;
%     symb1 = 571:600;
%     symb2 = 511:540;
% else
%     symb0 = 571:600;
%     symb1 = 541:570;
%     symb2 = 511:540;
% end
%     oldstim = stim(:,step:end);
% 
%     oldstim([symb0(1);symb1(1);symb2(1)] + 1,2:end) = oldstim([symb0(1);symb1(1);symb2(1)],1:end-1);
%     newstim = oldstim(:,2:end);
% end

function [lik,filtpower] = stimlikelihood(stim,emit_w)
    T = size(stim,2);
    numbins = size(stim,1);
    numstates = size(emit_w,1);

    filtpower = reshape(sum(reshape(repmat(emit_w,[1,1,T]),[numstates,numbins,T]) .* repmat(reshape(stim(:,:),1,numbins,T),[numstates,1,1]),2),numstates,T);
    lik = [ones(1,T); exp(filtpower)] ./ repmat(1+sum(exp(filtpower),1),[size(emit_w,1)+1,1]);
end

function [newemit,reallik,holdlik] = chooseNewStim(autostim,emit_w,p_each,emit)
    steps = length(emit) - length(autostim);
    reallik = zeros(size(autostim,1),size(autostim,3));
    holdlik = zeros(size(autostim,1),size(emit_w,1),size(autostim,3));
    for vv=1:size(autostim,1)
        % now compute the likelihood of observing the ACTUAL emission
        likpath = stimlikelihood(squeeze(autostim(vv,:,:)),emit_w);
        reallik(vv,:) = likpath(sub2ind(size(likpath),emit((steps+1):end)'+1,1:(length(emit)-steps)));
        holdlik(vv,:,:) = likpath(2:3,:);
    end
    
    newlik = reallik .* p_each;
    newlik = newlik ./ repmat(sum(newlik,1),[size(newlik,1),1]);
    newemit = sum(repmat(rand(1,size(autostim,3)),[size(newlik,1),1]) > cumsum(newlik),1) + 1;
end

function [newgrad, newvalue] = updateGrad(holdlik, reallik, newemit, autostim, grad, value, gamma, emit)
    steps = length(emit) - length(autostim);
    allgrad = [-squeeze(holdlik(sub2ind(size(holdlik),newemit,ones(1,length(newemit)),1:length(newemit)))); ...
                -squeeze(holdlik(sub2ind(size(holdlik),newemit,ones(1,length(newemit))+1,1:length(newemit))))];
    allgrad(1,emit((steps+1):end) == 1) = 1 + allgrad(1,emit((steps+1):end) == 1);
    allgrad(2,emit((steps+1):end) == 2) = 1 + allgrad(2,emit((steps+1):end) == 2);

    goodstim = zeros(size(autostim,2),size(autostim,3));
    goodgrad = zeros(size(grad,2),size(autostim,2),size(autostim,3));

    for vv=1:size(autostim,1)
        goodstim(:,newemit == vv) = squeeze(autostim(vv,:,newemit == vv));
        goodgrad(:,:,newemit == vv) = squeeze(grad(vv,:,:,newemit == vv));
    end

    newgrad = [(allgrad(1,:).*gamma((steps+1):end)) * goodstim';(allgrad(2,:).*gamma((steps+1):end)) * goodstim'];
    newgrad = newgrad + sum(goodgrad,3);

    newvalue = sum(value(sub2ind(size(value),newemit,1:length(value)))) + sum(gamma((steps+1):end).*log(reallik(sub2ind(size(reallik),newemit,1:length(reallik)))));
end