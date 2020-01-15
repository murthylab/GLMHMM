function [analoglik] = GLMHMM_AnalogLik(analog_emit_w,stim,analog_symb,numAnalogEmit)

    % create structures to store data in
    numAnalogParams = size(analog_symb{1},1);
    numtotalbins = size(analog_emit_w,3);
    numstates = size(analog_emit_w,1);
    for anum=numAnalogParams:-1:1
        residualBag{anum} = zeros(numstates,numAnalogEmit(anum));
        for trial=1:length(analog_symb)
            analog_residuals{trial} = zeros(size(analog_symb{trial},1),numstates,size(analog_symb{trial},2));
        end
    end

    % compute the residuals for the predictions in each state and each
    % analog variable
    aind = zeros(numAnalogParams,1);
    for trial=1:length(analog_symb)
        for anum=1:numAnalogParams
            goodemit = ~isnan(analog_symb{trial}(anum,:));
            T = sum(goodemit);

            thisstim = stim{trial}(:,goodemit);
            thisstim = repmat(reshape(thisstim,1,numtotalbins,T),[numstates,1,1]);

            prediction = sum(thisstim.*repmat(reshape(squeeze(analog_emit_w(:,anum,:)),numstates,numtotalbins,1),[1,1,T]),2);
            analog_residuals{trial}(anum,:,goodemit) = repmat(analog_symb{trial}(anum,goodemit),[numstates,1]) - reshape(prediction,numstates,T);

            residualBag{anum}(:,(aind(anum)+1):(aind(anum)+T)) = analog_residuals{trial}(anum,:,goodemit);
            if sum(isnan(analog_residuals{trial}(:))) > 0
                display('shit')
            end

            aind(anum) = aind(anum)+T;
        end
    end

    % compute the variance in the predictions
    analog_var = zeros(numstates,numAnalogParams);
    Z_analog = zeros(numstates,numAnalogParams);
    for anum=1:numAnalogParams
        for ss=1:numstates
            analog_var(ss,anum) = var(residualBag{anum}(ss,:));
        end
        Z_analog(:,anum) = real(sqrt(2*pi*analog_var(:,anum)));
    end

    % hack because a normalization constant below 1 can seriously screw
    % things up in the likelihood
    if any(Z_analog < 1)
        Z_analog = Z_analog ./ max(Z_analog(Z_analog < 1));
    end

    % now that we know the distribution that our predictions are coming from, compute the likelihoods
    for trial=1:length(analog_symb)
        analoglik{trial} = zeros(numAnalogParams,numstates,size(analog_symb{trial},2));
        for anum=1:numAnalogParams
            for ss=1:numstates
                analoglik{trial}(anum,ss,:) = 1./Z_analog(ss,anum) .* exp(-analog_residuals{trial}(anum,ss,:).^2./(2*analog_var(ss,anum)));
            end
        end
    end
end