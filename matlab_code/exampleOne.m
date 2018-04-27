function [outvars] = exampleOne(numbins,numstates,noiseSD)
    %% sample one: three inputs, two states, two discrete outputs, in state one input 1 is used in state two inputs 2-3 are used
    % try with both types of regularization

    tau = 4;
    bins = 30;
%     noiseSD = 0.1;
    numinputs = 3;
    totaltime = 10000;
    stimscale = 1;
    % numsamples = 20;
    numsamples = 5;
%     numstates = 2;
    numRealStates = 2;

    for ns=1:numsamples
        stim{ns} = makeStim(totaltime, bins, numinputs,stimscale) + randn(bins,totaltime,numinputs)*noiseSD;
        filt = makePeakedSampleFilter(tau, bins);

%         for ss=1:numRealStates
%             output(ss,:) = 1./(1+exp(stim{ns}(:,:,ss))'*filt') > 0.5;
%         end
        p1 = exp((stim{ns}(:,:,1))'*filt' + (stim{ns}(:,:,2))'*-filt');
        output(1,:) = p1./(1+p1) > 0.5;
        p1 = exp((stim{ns}(:,:,1))'*-filt' + (stim{ns}(:,:,2))'*filt');
        output(2,:) = p1./(1+p1) > 0.5;
%         output(1,:) = 1./(1+exp((stim{ns}(:,:,1))'*filt')) > 0.5;
%         output(2,:) = 1./(1+exp((stim{ns}(:,:,2))'*filt')) > 0.5;

        % generate transitions between states (use third input)
        p1 = exp(stim{ns}(:,:,end)'*filt');
        states{ns} = (p1./(1+p1) > 0.5) + 1;
        % generate output trace
        outputTrace{ns} = zeros(totaltime,1);
        for ss=1:numRealStates
            outputTrace{ns}(states{ns} == ss,1) = output(ss,states{ns} == ss);
        end

%         outputStim{ns} = squeeze(stim{ns}(1,:,:))';
%         outputStim{ns} = [squeeze(stim{ns}(2:numbins,1,:))', outputStim{ns}];
        outputStim{ns} = [stim{ns}(:,:,1); stim{ns}(:,:,2); stim{ns}(:,:,3); ones(1,totaltime)];
    end

    model_options.emit_lambda = 0;
    model_options.trans_lambda = 0;
%     model_options.L2smooth = 1;
%     model_options.smoothLambda = 1;
%     model_options.CVregularize = 1;
    
    model_options.L2smooth = 0;
    model_options.smoothLambda = 0;
    model_options.CVregularize = 0;

    model_options.numstates = numstates;
    
    features.emit_w = zeros(numstates, 1, numbins*3 + 1);
    features.emit_w(1,1,(1:numbins)) = filt;
    features.emit_w(1,1,numbins + (1:numbins)) = -filt;
    features.emit_w(2,1,(1:numbins)) = -filt;
    features.emit_w(2,1,numbins + (1:numbins)) = filt;
    
    features.trans_w = zeros(numstates, numstates, numbins*3 + 1);
    features.trans_w(1,2,numbins*2 + (1:numbins)) = filt;
    features.trans_w(2,1,numbins*2 + (1:numbins)) = -filt;
    
%     emit_w = rand(numstates,1,numinputs*numbins + 1);
%     trans_w = rand(numstates,numstates,numinputs*numbins + 1);
%     for ss=1:numstates
%         for ii=1:numinputs
%             for ee=1:1
%                 emit_w(ss,ee,(ii-1)*numbins + (1:numbins)) = exp(-(1:numbins)/30) * round(rand(1)*2 - 1);
%             end
%             for ss2=1:numstates
%                 trans_w(ss,ss2,(ii-1)*numbins + (1:numbins)) = exp(-(1:numbins)/30) * round(rand(1)*2 - 1);
%             end
%         end
%     end
    
%     features.emit_w = emit_w;
%     features.trans_w = trans_w;

    save tmp.mat states outputTrace outputStim features
    
    [outvars] = HMMGLMtrain9(outputTrace, features.emit_w, features.trans_w, outputStim, [], [], '', model_options);
%     [outvars] = fitGLMHMM(outputTrace,outputStim,numbins,model_options, features);
end
