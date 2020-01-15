function [outvars] = HMMGLMtrain9(symb, emit_w, trans_w, stim, analog_emit_w, analog_symb, outfilename, options)
% HMMGLMtrain - Fit a combined GLM/HMM
% symb - the discrete symbols emitted that are to be fit. These should be
% from 0..N-1 and in cells per trial to fit
% emit_w - the initial emission filter matrix
% (states,emissions-1,regressors)
% trans_w - the initial transition filter matrix (states,states,regressors)
% stim - the stimulus to be used for regressors {(regressors,time)}
% analog_emit_w - 
% analog_symb - 
% outfilename - filename to save each fitting iteration to
% options - many many fitting options

%% STEP ONE - DO ALL THE STUPID SETTING UP OF OPTIONS
% There's probably a better, more elegant way to do this

% dbstop if error
warning('off','MATLAB:dispatcher:nameConflict')

if ~exist('outfilename','var')
    outfilename = 'HMMGLM8_out.mat';
end

if ~exist('options','var')
    options.filler = 0;
end
if ~isfield(options,'useASD')
    options.useASD = 1;
end
if ~isfield(options,'filterOffset')
    options.filterOffset = 1;
end
if ~isfield(options,'L2smooth')
    options.L2smooth = 0;
end
if ~isfield(options,'smoothLambda')
    options.smoothLambda = 0;
end
if ~isfield(options,'numFilterBins')
    options.numFilterBins = 30;
end
if ~isfield(options,'trans_lambda')
    options.trans_lambda = 0;
end
if ~isfield(options,'emit_lambda')
    options.emit_lambda = 0;
end
if ~isfield(options,'AR_lambda')
    options.AR_lambda = -1;
end
if ~isfield(options,'ARvec')
    options.stimvec = [1:510 631];
    options.ARvec = 511:630;
end
if ~isfield(options,'evaluate')
    options.evaluate = 0;
end
if ~isfield(options,'generate')
    options.generate = 0;
else
    options.evaluate = 1;
end
if ~isfield(options,'analog_flag')
    options.analog_flag = false;
end
if ~isfield(options,'anneal_lambda')
    options.anneal_lambda = false;
end
if ~isfield(options,'autoanneal')
    options.autoanneal = false;
else
    if ~isfield(options,'autoanneal_vec')
        options.autoanneal_vec = [.01,.025,.05,.075,.1,.25,.5,.75,1];
        options.autoanneal_schedule = [1,5,7,8,9,10,11,12,13,14,15,16];
    end
end
if ~isfield(options,'GLMtransitions')
    options.GLMtransitions = true;
end
if ~isfield(options,'fitEmissions')
    options.fitEmissions = true;
end
if ~isfield(options,'GLMemissions')
    options.GLMemissions = true;
end
if ~isfield(options,'getErrorBars')
    options.getErrorBars = false;
end
if ~isfield(options,'numsteps')
    options.numsteps = 1;
end
if ~isfield(options,'numsamples')
    options.numsamples = 1;
end
if ~isfield(options,'trainBins')
    options.trainBins = [];
end
if ~isfield(options,'CVregularize')
    options.CVregularize = 0;
    options.regularSchedule = 1;
    options.regularScheduleInd = 1;

    options.trainData = 1:length(stim);
    options.testData = [];
else
    if options.CVregularize
        if ~isfield(options,'regularSchedule')
            options.regularSchedule = logspace(-4,1,10);
            options.regularScheduleInd = 8;
        end

        rdata = randperm(length(stim));
        options.testData = rdata(1:ceil(.25*length(rdata)));
        options.trainData = rdata((ceil(.25*length(rdata))+1):end);
    else
        options.CVregularize = 0;
        options.regularSchedule = 1;
        options.regularScheduleInd = 1;
        
        options.trainData = 1:length(stim);
        options.testData = [];
    end
end
if ~isfield(options,'crossvalidate')
    options.crossvalidate = 0;
    if ~options.CVregularize
        options.trainData = 1:length(stim);
        options.testData = [];
    end
else
    if options.crossvalidate
        rdata = randperm(length(stim));
        options.testData = rdata(1:ceil(.25*length(rdata)));
        options.trainData = rdata((ceil(.25*length(rdata))+1):end);    
    else
        options.crossvalidate = 0;
        options.trainData = 1:length(stim);
        options.testData = [];
    end
end

if ~isfield(options,'addFilters')
    options.addFilters = 0;
else
    options.initial_emit_w = emit_w;
    options.initial_trans_w = trans_w;
    options.initial_analog_emit_w = analog_emit_w;
end

if isempty(symb)
    options.symbExists = false;
else
    options.symbExists = true;
end


% Add all subdirectories to path - NOTE that I need to change this for
% broader consumption (assuming not everyone has access to my Dropbox)
if ispc
    addpath('C:\Users\adamjc\Dropbox\MurthyLab\Utilities\helpers\')
    pdir = 'C:/Users/adamjc/Dropbox/MurthyLab/OpenProjects/GenerativeModel/';
elseif ismac
    addpath('~/Dropbox/MurthyLab/Utilities/helpers/')
    pdir = '~/Dropbox/MurthyLab/OpenProjects/GenerativeModel/';
else
    if exist('/jukebox/','dir')
        addpath('/jukebox/murthy/adamjc/Utilities/helpers/')
        pdir = '/jukebox/murthy/adamjc/Utilities/GLM-HMM/';
    else
        addpath('/tigress/adamjc/Utilities/helpers/')
        pdir = '/tigress/adamjc/Utilities/GLM-HMM/';
    end
end
addpath([pdir 'minFunc']);
addpath([pdir 'minFunc/compiled']);
% addpath('rr');
% addpath('rr/tools');
addpath([pdir 'fastasd/code']);
addpath([pdir 'fastasd/code/code_ridgeRegress']);
addpath([pdir 'fastasd/code/tools_dft']);
addpath([pdir 'fastasd/code/tools_kron']);
addpath([pdir 'fastasd/code/tools_optim']);


% We expect the symbols to be in a cell; if they're not, make them a cell
if options.symbExists
    if ~iscell(symb)
        symb2 = symb;
        clear symb
        symb{1} = symb2;
    end
else
    for trial=1:length(analog_symb)
        symb{trial} = [];
    end
end

% figure out from the data how many symbols there are to fit (though, really, this should be the same as the second
% dimension of emit_w + 1 so I'm not sure why I'm doing it this way...)
allsymb = [];
for i=1:length(symb)
    allsymb = [allsymb, unique(symb{i}(:))'];
end
allsymb = unique(allsymb);
numsymb = length(allsymb);

if ~isfield(options,'maxiter')
    maxiter = 1000;
else
    maxiter = options.maxiter;
end
loglik = zeros(maxiter+1,1);
loglik(1) = -10000000;
thresh = 1e-4;

numstates = max(max(size(emit_w,1),size(analog_emit_w,1)),size(trans_w,1));
numemissions = size(emit_w,2);
numtotalbins = max(max(size(emit_w,3),size(trans_w,3)),size(analog_emit_w,3));

if options.analog_flag
    numAnalogParams = size(analog_symb{1},1);
    numAnalogEmit = zeros(numAnalogParams,1);
else
    numAnalogParams = 0;
end

for trial=1:max(length(symb),length(analog_symb))
    prior{trial} = ones(numstates,1)./numstates;     % is this good?!?!
    if options.analog_flag
        for anum=1:numAnalogParams
            numAnalogEmit(anum) = numAnalogEmit(anum) + sum(~isnan(analog_symb{trial}(anum,:)));
        end

        analogResiduals{numAnalogParams} = [];
        analog_prediction{length(symb)}(numAnalogParams).pred = [];
        gamma{trial} = ones(numstates,size(analog_symb{trial},2));
    else
        gamma{trial} = ones(numstates,size(symb{trial},2));
    end
    gamma{trial} = gamma{trial} ./ repmat(sum(gamma{trial},1),[numstates,1]);
end

% First we need to know the likelihood of seeing each symbol given the filters
% as well as the likelihood of seeing a transition from step to step

xi{max(length(symb),length(analog_symb))} = [];
effectiveInd = 1;
lastTry = false;
for trial=1:max(length(symb),length(analog_symb))
    if options.symbExists
        [symblik{trial}] = GLMHMM_SymbLik(emit_w,stim{trial},symb{trial});
    end
    [transition{trial}] = GLMHMM_TransLik(trans_w,stim{trial});
end
if options.analog_flag
    [analoglik] = GLMHMM_AnalogLik(analog_emit_w,stim,analog_symb,numAnalogEmit);
end
for ind = 1:maxiter
    display(['fitting iteration ' num2str(ind)])

    for trial=1:max(length(symb),length(analog_symb))
        % maybe first compute likelihoods for the symbols?
        if options.analog_flag && options.symbExists
            if numstates == 1       % MATLAB is stupid about how it shapes vectors
                emitLikeli = symblik{trial} .* squeeze(prod(analoglik{trial},1))';
            else
                emitLikeli = symblik{trial} .* squeeze(prod(analoglik{trial},1));
            end
        elseif options.symbExists
            emitLikeli = symblik{trial};
        elseif options.analog_flag
            if numstates == 1
                emitLikeli = squeeze(prod(analoglik{trial},1))';
            else
                emitLikeli = squeeze(prod(analoglik{trial},1));
            end
        end
        emitLikeli(emitLikeli < eps(0)*1e3) = eps(0)*1e3;
        [gamma{trial},xi{trial},prior{trial}] = computeTrialExpectation(prior{trial},emitLikeli,transition{trial});
    end

%% gradient descent for the emission filter

if options.symbExists && options.fitEmissions
    display('fitting categorical emission filters')

    clear newstim
    if options.crossvalidate || options.CVregularize
        if options.CVregularize
            CVschedule = options.regularSchedule(max(options.regularScheduleInd-1,1):min(options.regularScheduleInd+1,length(options.regularSchedule)));
        else
            CVschedule = [options.smoothLambda];
        end
        CVind = 0;

        for thislambda = CVschedule
            CVind = CVind + 1;
            % check if we are doing smoothing regularization or Tikhonov
            % segment data into random subsets for cross validation
            if options.smoothLambda == -1
                options.transLambda = thislambda;
                options.emitLambda = thislambda;
            else
                options.smoothLambda = thislambda;
            end

            if options.symbExists
                % fitEmissionFilters should take into account the testData
                % field...
                [newEmit_w{CVind}, pgd_lik, pgd_prob, pgd_prob2, pgd_prob3, pgd_prob4] = fitEmissionFilters(stim,symb,gamma,xi,emit_w,options);
            else
                pgd_lik = 0;
            end

            [newTrans_w{CVind}, tgd_lik] = fitTransitionFilters(stim,symb,gamma,xi,trans_w,options);

            if options.analog_flag
                [newAnalog_emit_w{CVind}, newAnalog_emit_std{CVind},arcorr,arcorr2] = fitAnalogFilters(stim, analog_symb, analog_emit_w, options);
            end
            
            if options.analog_flag
                for i=1:length(options.testData)
                    testsymb{i} = symb{options.testData(i)};
                    testanalogsymb{i} = symb{options.testData(i)};
                    teststim{i} = stim{options.testData(i)};
                end
                [testEmitLik{CVind},testTransLik{CVind},testAnalogLik{CVind},testGamma{CVind},testXi{CVind}] = HMMGLMLikelihoods(outparams.emit, newEmit_w{CVind}, newTrans_w{CVind}, outparams.stim, newAnalog_emit_w{CVind}, outparams.analog_symb, model_options);
            else
                for i=1:length(options.testData)
                    testsymb{i} = symb{options.testData(i)};
                    teststim{i} = stim{options.testData(i)};
                end
                [testEmitLik{CVind},testTransLik{CVind},testAnalogLik{CVind},testGamma{CVind},testXi{CVind}] = HMMGLMLikelihoods(testsymb, newEmit_w{CVind}, newTrans_w{CVind}, teststim, [], [], options);
            end
    
            testFullAnalogEmitLik = 0;
            testFullEmitLik  = 0;
            testFullTransLik  = 0;
            testFullBasicLik = zeros(length(options.testData),1);
            for i=1:length(options.testData)
                gamma{i}(gamma{i}(:,1) == 0,1) = eps(0);

                if options.symbExists
                    testFullEmitLik = testFullEmitLik + -mean(sum(testGamma{CVind}{i} .* log(testEmitLik{CVind}{i})));
                    testFullTransLik = testFullTransLik + -mean(sum(sum(testXi{CVind}{i} .* log(testTransLik{CVind}{i}(:,:,2:end)))));
                end

                if options.analog_flag
                    analogprod = prod(testAnalogLik{CVind}{i},1);
                    analogprod(analogprod == 0) = eps(0);
                    if numstates == 1
                        testFullAnalogEmitLik = testFullAnalogEmitLik + -mean(sum(testGamma{CVind}{i} .* log(squeeze(analogprod)')));
                    else
                        testFullAnalogEmitLik = testFullAnalogEmitLik + -mean(sum(testGamma{CVind}{i} .* log(squeeze(analogprod))));
                    end
                end

                testFullBasicLik(i) = -sum(testGamma{CVind}{i}(:,1).*log(testGamma{CVind}{i}(:,1)));
            end
            testLogLik(CVind) = sum(testFullBasicLik) + testFullEmitLik + testFullTransLik + testFullAnalogEmitLik;
        end

        if options.CVregularize
            CVinds = max(options.regularScheduleInd-1,1):min(options.regularScheduleInd+1,length(options.regularSchedule));
            goodInd = find(testLogLik == min(testLogLik));

            if options.symbExists
                emit_w = newEmit_w{goodInd};
                trans_w = newTrans_w{goodInd};
            end
            if options.analog_flag
                analog_emit_w = newAnalog_emit_w{goodInd};
                analog_emit_std = newAnalog_emit_std{goodInd};
            end

            thislambda = options.regularSchedule(CVinds(goodInd));
            options.regularScheduleInd = CVinds(goodInd);
            if options.smoothLambda == -1
                options.transLambda = thislambda;
                options.emitLambda = thislambda;
            else
                options.smoothLambda = thislambda;
            end
        else
            goodInd = 1;
        end

        if options.crossvalidate
            outvars(ind).lambda = thislambda;
            outvars(ind).loglik_CV = testLogLik(goodInd);
            outvars(ind).loglik_CVlambda = CVschedule;
            outvars(ind).loglik_CVall = testLogLik;
        end
        
    else
        for trial=1:length(stim)
            if options.GLMemissions
                %             if isempty(options.trainBins)
                newstim{trial}.data = stim{trial};
                newstim{trial}.numtotalbins = numtotalbins;
                %             else
                %                 newstim{trial}.data = stim{trial}(options.trainBins,:);
                %                 newstim{trial}.numtotalbins = length(options.trainBins);
                %             end
                
            else
                newstim{trial}.data = stim{trial}(end,:);
                newstim{trial}.numtotalbins = 1;
            end
            newstim{trial}.xi = xi{trial};
            newstim{trial}.gamma = gamma{trial};
            newstim{trial}.emit = symb{trial};
            
            newstim{trial}.numstates = numemissions;
        end
        
        tmp_pgd = zeros(numstates,1);
        tmp_pgd2 = zeros(numstates,1);
        tmp_pgd3 = zeros(numstates,numemissions+1,numemissions+1);
        tmp_pgd4 = zeros(numstates,numemissions+1,numemissions+1);
        tmp_pgd_lik = zeros(numstates,1);
        hessdiag_emit = zeros(numstates,numemissions,numtotalbins);
        for i=1:numstates
            if options.numsteps == 1
                %                 if isempty(options.trainBins)
                outweights = minFunc(@(x) emitLearningFun(x, newstim, i, options), reshape(squeeze(emit_w(i,:,:))',size(emit_w,2)*size(emit_w,3),1));
                emit_w(i,:,:) = reshape(outweights,size(emit_w,3),size(emit_w,2))';
                %                 else
                %                     outweights = minFunc(@(x) emitLearningFun(x, newstim, i, options), reshape(squeeze(emit_w(i,:,options.trainBins))',size(emit_w,2)*length(options.trainBins),1));
                %                     emit_w(i,:,options.trainBins) = reshape(outweights,length(options.trainBins),size(emit_w,2))';
                %                 end
            else
                outweights = minFunc(@(x) emitMultistepLearningFun(x, newstim, i, options), reshape(squeeze(emit_w(i,:,:))',size(emit_w,2)*size(emit_w,3),1));
                emit_w(i,:,:) = reshape(outweights,size(emit_w,3),size(emit_w,2))';
            end
        end
        pgd_lik = sum(tmp_pgd);
        
        if ~isempty(options.trainBins)
            for trial = 1:length(stim)
                newstim{trial}.data = stim{trial};
            end
        end
        
        for i=1:numstates
            [tmp_pgd(i),hessd] = emitLearningStats(reshape(squeeze(emit_w(i,:,:))',size(emit_w,2)*size(emit_w,3),1), newstim, i, options);
            hessdiag_emit(i,:,:) = reshape(hessd,size(hessdiag_emit,3),size(hessdiag_emit,2))';
            
            [tmp_pgd_lik(i),tmp_pgd(i),tmp_pgd2(i), tmp_pgd3_2, tmp_pgd4_2] = emitLikelihood(reshape(squeeze(emit_w(i,:,:))',size(emit_w,2)*size(emit_w,3),1), newstim, i);
            tmp_pgd3(i,:,:) = tmp_pgd3_2;
            tmp_pgd4(i,:,:) = tmp_pgd4_2;
        end
        pgd_prob = sum(tmp_pgd);
        pgd_prob2 = sum(tmp_pgd2);
        pgd_prob3 = squeeze(sum(tmp_pgd3,1));
        pgd_prob4 = squeeze(sum(tmp_pgd4,1));
    end
    
    

else
    pgd_lik = 0;
    pgd_prob = 0;
end
%% gradient descent for the transition filter

    display('fitting transition filters')

    for trial=1:length(stim)
        newstim{trial}.numstates = numstates;
        if options.GLMtransitions
            if isempty(options.trainBins)
                newstim{trial}.data = stim{trial};
                newstim{trial}.numtotalbins = numtotalbins;
            else
                newstim{trial}.data = stim{trial}(options.trainBins,:);
                newstim{trial}.numtotalbins = length(options.trainBins);
            end
        else
            newstim{trial}.data = stim{trial}(end,:);
            newstim{trial}.numtotalbins = 1;
        end

        newstim{trial}.xi = xi{trial};
        newstim{trial}.gamma = gamma{trial};
    end
    tmp_tgd = zeros(numstates,1);
    if options.evaluate == 1
        for i=1:numstates
            tmp_tgd(i) = transLearningFun(reshape(squeeze(trans_w(i,:,:))',size(trans_w,2)*size(trans_w,3),1), newstim, i, options);
        end
        tgd_lik = sum(tmp_tgd);
    else
        hessdiag_trans = zeros(numstates,numstates,numtotalbins);
        for i=1:numstates
            if isempty(options.trainBins)
                outweights = minFunc(@(x) transLearningFun(x, newstim, i, options), reshape(squeeze(trans_w(i,:,:))',size(trans_w,2)*size(trans_w,3),1));
                trans_w(i,:,:) = reshape(outweights,size(trans_w,3),size(trans_w,2))';
            else
                outweights = minFunc(@(x) transLearningFun(x, newstim, i, options), reshape(squeeze(trans_w(i,:,options.trainBins))',size(trans_w,2)*length(options.trainBins),1));
                trans_w(i,:,options.trainBins) = reshape(outweights,length(options.trainBins),size(trans_w,2))';
            end
        end
        
        if ~isempty(options.trainBins)
            for trial = 1:length(stim)
                newstim{trial}.data = stim{trial};
            end
        end

        for i=1:numstates
            tmp_tgd(i) = transLearningFun(reshape(squeeze(trans_w(i,:,:))',size(trans_w,2)*size(trans_w,3),1), newstim, i, options);

            if (numstates > 1)
                [tmp_pgd(i),hessd] = transLearningStats(reshape(squeeze(trans_w(i,:,:))',size(trans_w,2)*size(trans_w,3),1), newstim, i, options);
                hessdiag_trans(i,:,:) = reshape(hessd,size(hessdiag_trans,3),size(hessdiag_trans,2))';
            else
                hessdiag_trans = 0;
            end
        end
        tgd_lik = sum(tmp_tgd);
    end
%% don't need to gradient descent for straight-up regressions, just need to regress!
% we're just doing weighted least squares here https://en.wikipedia.org/wiki/Least_squares#Weighted_least_squares
% options.evaluate = 1;

% need to see how much data we need to accurately reconstruct these filters
% and why the smooth asd is failing so badly so often

    if options.analog_flag
        display('fitting analog emission filters')

        if options.evaluate == 1
            for anum=1:numAnalogParams
                for trial=1:length(stim)
                    newstim{trial}.symb = analog_symb{trial}(anum,:);
                    newstim{trial}.goodemit = ~isnan(analog_symb{trial}(anum,:));
                    newstim{trial}.numtotalbins = numtotalbins;
                    newstim{trial}.data = stim{trial};
                end
                [thesestim,thesesymb,thesegamma] = collectWLSInfo(newstim);
                for states = 1:numstates
                    arcorr(states,anum) = wcorr(thesestim*squeeze(analog_emit_w(states,anum,:)),thesesymb,thesegamma(states,:)');
                end
                arcorr2(anum) = sum(mean(thesegamma,2).*arcorr(:,anum));
            end
            analog_emit_std = 0;
        else
            for anum=1:numAnalogParams
                display(['fitting filter ' num2str(anum) '/' num2str(numAnalogParams)]);
                for trial=1:length(stim)
                    newstim{trial}.symb = analog_symb{trial}(anum,:);
                    newstim{trial}.goodemit = ~isnan(analog_symb{trial}(anum,:));
                    newstim{trial}.data = stim{trial};
                    newstim{trial}.numtotalbins = numtotalbins;
                end
                [thesestim,thesesymb,thesegamma] = collectWLSInfo(newstim);

                % if more than this, loop until we have gone through all of
                % them. How to deal with eg ~1k over this max? overlapping
                % subsets? Could just do eg 4 subsamples, or however many
                % depending on amount >15k
                maxgoodpts = 15000;
                numAnalogIter = ceil(size(thesestim,1) ./ maxgoodpts);
                if numAnalogIter > 1
                    analogOffset = (size(thesestim,1) - maxgoodpts) ./ (numAnalogIter - 1);
                    % (numAnalogIter - 1) * offset + maxgoodpts = size(thesestim,1)
                end

                randomizedstim = randperm(size(thesestim,1));
                if numAnalogIter > 1
                    iterstim = zeros(numAnalogIter,2);
                    for nai=1:numAnalogIter
                        iterstim(nai,:) = floor(analogOffset*(nai-1)) + [1,maxgoodpts];
                    end
                else
                    iterstim = [1,size(thesestim,1)];
                end

                aew = zeros(numAnalogIter,numstates,size(analog_emit_w,3));
                aestd = zeros(numAnalogIter,numstates,size(analog_emit_w,3));
                iterweight = zeros(numAnalogIter,numstates);
                for nai=1:numAnalogIter
                    usestim = randomizedstim(iterstim(nai,1):iterstim(nai,2));
                    for states = 1:numstates
                        if options.useASD
                            if rem(size(thesestim,2),options.numFilterBins) == 1
                                [outweights,ASDstats] = fastASDweighted_group(thesestim(usestim,:),thesesymb(usestim),thesegamma(states,usestim),[ones(round(size(thesestim,2)/options.numFilterBins),1)*options.numFilterBins;1],2);
                            else
                                [outweights,ASDstats] = fastASDweighted_group(thesestim(usestim,:),thesesymb(usestim),thesegamma(states,usestim),[ones(round(size(thesestim,2)/options.numFilterBins),1)*options.numFilterBins],2);
                            end
                            aew(nai,states,:) = outweights;
                            aestd(nai,states,:) = ASDstats.Lpostdiag;
                        else
                            [outweights, outstd] = weightedLSbystate(thesestim(usestim,:), thesesymb(usestim), thesegamma(states,usestim), 10);
                            aew(nai,states,:) = outweights;
                            aestd(nai,states,:) = outstd;
                        end
                        iterweight(nai,states) = sum(thesegamma(states,usestim));
                        arcorr(states,anum) = 0;
%                         [outweights,ASDstats] = fastASDweighted_group(thesestim(usestim,:),thesesymb(usestim),thesegamma(states,usestim),[ones(round(size(thesestim,2)/30),1)*30],2);
%                         [outweights, outstd] = weightedLSbystate(thesestim(usestim,:), thesesymb(usestim), thesegamma(states,usestim), 10);
                        
%                         aestd(nai,states,:) = outstd;
                        
                    end
                end
                for states = 1:numstates
                    analog_emit_w(states,anum,:) = sum(aew(:,states,:) .* repmat(iterweight(:,states) ./ sum(iterweight(:,states)),[1,1,size(analog_emit_w,3)]),1);
                    analog_emit_std(states,anum,:) = sum(aestd(:,states,:) .* repmat(iterweight(:,states) ./ sum(iterweight(:,states)),[1,1,size(analog_emit_w,3)]),1);
                end

                arcorr(states,anum) = 0;
                arcorr2(anum) = 0;
            end
        end
    end

% calculate analog likelihood...

%%
    analogemit_lik = 0;
    trans_lik = 0;
    emit_lik = 0;
    if options.analog_flag
        [analoglik] = GLMHMM_AnalogLik(analog_emit_w,stim,analog_symb,numAnalogEmit);
    end
    for trial=1:max(length(symb),length(analog_symb))
        if options.symbExists
            [symblik{trial}] = GLMHMM_SymbLik(emit_w,stim{trial},symb{trial});
            [transition{trial}] = GLMHMM_TransLik(trans_w,stim{trial});
            emit_lik = emit_lik + -sum(sum(gamma{trial} .* log(symblik{trial})));
            trans_lik = trans_lik + -sum(sum(sum(xi{trial} .* log(transition{trial}(:,:,2:end)))));
        end

        if options.analog_flag
            analogprod = prod(analoglik{trial},1);
            analogprod(analogprod == 0) = eps(0);
            if numstates == 1
                analogemit_lik = analogemit_lik + -sum(sum(gamma{trial} .* log(squeeze(analogprod)')));
            else
                analogemit_lik = analogemit_lik + -sum(sum(gamma{trial} .* log(squeeze(analogprod))));
            end
        end
    end

    % log likelihood: sum(gamma(n) * log) + tgd_lik + pgd_lik
    for i=1:length(symb)
        gamma{i}(gamma{i}(:,1) == 0,1) = eps(0);
        basic_lik(i) = -sum(gamma{i}(:,1).*log(gamma{i}(:,1)));
    end
    loglik(ind+1) = sum(basic_lik) + emit_lik + trans_lik + analogemit_lik;
%     loglik(ind+1) = tgd_lik + pgd_lik;

    if options.generate
        outvars(ind).output = output;
        outvars(ind).output_anal = output_anal;
    end
    if options.analog_flag
        outvars(ind).analog_emit_w = analog_emit_w;
        outvars(ind).analog_emit_std = sqrt(analog_emit_std);
        outvars(ind).arcorr = arcorr;
        outvars(ind).arcorr2 = arcorr2;
        outvars(ind).agd_lik = analogemit_lik;
        outvars(ind).analogemit_lik = analogemit_lik;
    end
    if options.getErrorBars
        outvars(ind).analog_emit_std = analog_emit_std;
        outvars(ind).arcorr = arcorr;
        outvars(ind).arcorr2 = arcorr2;
        outvars(ind).emit_std = emit_std;
        outvars(ind).trans_std = trans_std;
        outvars(ind).numFolds = folds;
        if options.analog_flag
            outvars(ind).analogFolds = 0;
        end
    end

    if options.symbExists && options.fitEmissions
        outvars(ind).emit_lik = emit_lik;
        outvars(ind).pgd_prob = pgd_prob;
        outvars(ind).pgd_prob2 = pgd_prob2;
        outvars(ind).pgd_prob3 = pgd_prob3;
        outvars(ind).pgd_prob4 = pgd_prob4;
        if exist('hessdiag_emit','var')
            outvars(ind).emit_std = sqrt(hessdiag_emit);
        end
    end

    outvars(ind).emit_w = emit_w;
    outvars(ind).trans_w = trans_w;
    if exist('hessdiag_trans','var')
        outvars(ind).trans_std = sqrt(hessdiag_trans);
    end
    outvars(ind).trans_lik = trans_lik;
    outvars(ind).tgd_lik = tgd_lik;
    outvars(ind).pgd_lik = pgd_lik;
    outvars(ind).loglik = loglik(2:ind+1);
%     outvars(ind).gamma = gamma;
    outvars(ind).trans_lambda = options.trans_lambda;
    outvars(ind).emit_lambda = options.emit_lambda;
    outvars(ind).smoothLambda = options.smoothLambda;
    save(outfilename,'outvars','options','-v7.3');

    display(['log likelihood: ' num2str(loglik(ind+1))]);
%     if (abs(loglik(ind+1)-loglik(ind)) < thresh) || (loglik(ind+1)-loglik(ind) > 0)
    % now do this for not just the loglik but *each* of the likelihoods
    % individually
    if (abs(loglik(ind+1)-loglik(ind))/abs(loglik(ind)) < thresh)
%     if (abs(loglik(ind+1)-loglik(ind)) < thresh) && ind > 20

        if lastTry
            loglik = loglik(2:ind+1);

            for anum=1:numAnalogParams
                display(['fitting filter ' num2str(anum) '/' num2str(numAnalogParams)]);
                for trial=1:length(stim)
                    newstim{trial}.symb = analog_symb{trial}(anum,:);
                    newstim{trial}.goodemit = ~isnan(analog_symb{trial}(anum,:));
                end
                [thesestim,thesesymb,thesegamma] = collectWLSInfo(newstim);
                
                % if more than this, loop until we have gone through all of
                % them. How to deal with eg ~1k over this max? overlapping
                % subsets? Could just do eg 4 subsamples, or however many
                % depending on amount >15k
                maxgoodpts = 15000;
                numAnalogIter = ceil(size(thesestim,1) ./ maxgoodpts);
                if numAnalogIter > 1
                    analogOffset = (size(thesestim,1) - maxgoodpts) ./ (numAnalogIter - 1);
                    % (numAnalogIter - 1) * offset + maxgoodpts = size(thesestim,1)
                end
                
                randomizedstim = randperm(size(thesestim,1));
                if numAnalogIter > 1
                    iterstim = zeros(numAnalogIter,2);
                    for nai=1:numAnalogIter
                        iterstim(nai,:) = floor(analogOffset*(nai-1)) + [1,maxgoodpts];
                    end
                else
                    iterstim = [1,size(thesestim,1)];
                end
                
                aew = zeros(numAnalogIter,numstates,size(analog_emit_w,3));
                aestd = zeros(numAnalogIter,numstates,size(analog_emit_w,3));
                for nai=1:numAnalogIter
                    usestim = randomizedstim(iterstim(nai,1):iterstim(nai,2));
                    for states = 1:numstates
                        [outweights,ASDstats] = fastASDweighted_group(thesestim(usestim,:),thesesymb(usestim),thesegamma(states,usestim),[ones(round(size(thesestim,2)/30),1)*30;1],2);
                        aew(nai,states,:) = outweights;
                        aestd(nai,states,:) = ASDstats.Lpostdiag;

                        iterweight(nai,states) = sum(thesegamma(states,usestim));
                        arcorr(states,anum) = 0;
                    end
                end
                for states = 1:numstates
                    analog_emit_w(states,anum,:) = sum(aew(:,states,:) .* repmat(iterweight(:,states) ./ sum(iterweight(:,states)),[1,1,size(analog_emit_w,3)]),1);
                    analog_emit_std(states,anum,:) = sum(aestd(:,states,:) .* repmat(iterweight(:,states) ./ sum(iterweight(:,states)),[1,1,size(analog_emit_w,3)]),1);
                end
            end

            outvars(ind).analog_emit_w_ASD = analog_emit_w;
            outvars(ind).analog_emit_std_ASD = analog_emit_w;

            display('change in log likelihood below threshold!');
            break;
        else
            lastTry = true;
            if effectiveInd < 4
                effectiveInd = 5;   % since the regularization schedule starts here...
            else
                effectiveInd = effectiveInd + 1;
            end
        end
    else
        effectiveInd = effectiveInd + 1;
        lastTry = false;
    end

    if (options.autoanneal)
        lambda = regularizationSchedule(effectiveInd);
        options.trans_lambda = lambda;
        options.emit_lambda = lambda;
    end

    if options.evaluate == 1 || options.getErrorBars
        break;
    end
end

display('FINISHED')



end