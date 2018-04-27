import numpy as np
import matplotlib.pyplot as plt

def emitLearningFun(emit_w, stim, statenum, options):

    numstates = stim[0]['numstates']
    numbins = stim[0]['numtotalbins']
    # states x bins
    emit_w = np.reshape(emit_w,(numbins,numstates)).T;

    allgrad = np.zeros((numstates, numbins));
    allvalue = 0;
    totalT = 0;

    thisLambda = options['emit_lambda']

    # find out how many data points we are dealing with so that we can normalize
    # (I don't think we actually need to do this but it helps keep regularization values consistent from fit to fit)
    for trial in range(len(stim)):
        totalT = totalT + stim[trial]['data'].shape[0]

    for trial in range(len(stim)):
        T = stim[trial]['data'].shape[1]
        # convert into states x bins x time and sum across bins
        filtpower = np.reshape(np.sum(np.reshape(np.tile(emit_w,(1,1,T)),(numstates,numbins,T)) * np.tile(np.reshape(stim[trial]['data'],(1,numbins,T)),(numstates,1,1)),axis=1),(numstates,T));
        # now filtpower is states x time
        # filtpower is the filter times the stimulus

        # build up the value function:
        # gamma * log(exp(filtpower) / (1 + sum(exp(filtpower))))
        # = gamma * filtpower - gamma * log(1 + sum(exp(filtpower)))
        #
        # gradient is then:
        # gamma * (1|emission - exp(filtpower) / (1+sum(exp(filtpower)))) * stim
        value = stim[trial]['gamma'][statenum,:] * -np.log(1 + np.sum(np.exp(filtpower),axis=0))
        tgrad = -np.exp(filtpower) / np.tile(1 + np.sum(np.exp(filtpower),axis=0),(numstates,1))

        for i in range(filtpower.shape[0]):
            tgrad[i,stim[trial]['emit'] == i] = 1 + tgrad[i,stim[trial]['emit'] == i];
            value[stim[trial]['emit'] == i] = value[stim[trial]['emit'] == i] + stim[trial]['gamma'][statenum,stim[trial]['emit'] == i] * filtpower[i,stim[trial]['emit'] == i]

        value = np.sum(value);
        if np.any(np.isnan(value)):
            print('ugh')

        tgrad = tgrad * np.tile(stim[trial]['gamma'][statenum,:],(numstates,1))
        tgrad = np.sum(np.tile(np.reshape(tgrad,(numstates,1,T)),(1,numbins,1)) * np.tile(np.reshape(stim[trial]['data'],(1,numbins,T)),(numstates,1,1)),axis=2)

        allgrad = allgrad + tgrad
        allvalue = allvalue + value
    

    gradRegularization = 0;
    valueRegularization = 0;
    # oops guess I never implemented the regularization in python - but should be pretty straightforward

    # if options['L2smooth']:
    #     Dx1 = spdiags(ones(size(emit_w,2)-1,1)*[-1 1],0:1,size(emit_w,2)-1-1,size(emit_w,2)-1); 
    #     Dx = Dx1'*Dx1; # computes squared diffs

    #     for fstart=(options.numFilterBins+1):options.numFilterBins:(size(emit_w,2)-1)
    #         Dx(fstart,fstart) = 1;
    #         Dx(fstart-1,fstart-1) = 1;
    #         Dx(fstart-1,fstart) = 0;
    #         Dx(fstart,fstart-1) = 0;
    #     D = blkdiag(Dx,0);


    #     if options.AR_lambda ~= -1
    #         if length(options.smoothLambda) == 1
    #             options.smoothLambda = repmat(options.smoothLambda(1),[size(emit_w,1),size(emit_w,2)]);
    #             options.smoothLambda(:,options.ARvec) = options.AR_lambda;
            
    #         gradRegularization = gradRegularization + options.smoothLambda .* (D * emit_w')';
    #         valueRegularization = valueRegularization + sum(sum(((options.smoothLambda/2) .* (D * emit_w(:,:)')').^2));
    #     else
    #         gradRegularization = gradRegularization + options.smoothLambda .* (D * emit_w')';
    #         valueRegularization = valueRegularization +  sum(sum(((options.smoothLambda/2) .* (D * emit_w(:,:)')').^2));
    # if thisLambda ~= 0
    #     if options.AR_lambda ~= -1
    #         gradRegularization = gradRegularization + [thisLambda * emit_w(:,options.stimvec), options.AR_lambda * emit_w(:,options.ARvec)];
    #         valueRegularization = valueRegularization + (thisLambda/2) * sum(sum(emit_w(:,options.stimvec).^2)) + (options.AR_lambda/2) * sum(sum(emit_w(:,options.ARvec).^2));
    #     else
    #         gradRegularization = gradRegularization + thisLambda * emit_w(:,:);
    #         valueRegularization = valueRegularization + (thisLambda/2) * sum(sum(emit_w(:,:).^2));

    allgrad = -allgrad/totalT + gradRegularization;
    allvalue = -allvalue/totalT + valueRegularization;

    if np.any(np.isnan(allgrad)) or np.any(np.isnan(allvalue)):
        print('WTF! SOMETHING BAD HAPPENED! ISNAN! OH')
    

    # if nargout > 1:
    #     allgrad = reshape(allgrad',size(allgrad,1)*size(allgrad,2),1);

    # print(allgrad.shape)
    # print(allvalue.shape)
    return allvalue, np.ndarray.flatten(allgrad)




def transLearningFun(trans_w, stim, statenum, options):
    # trans_w are the weights that we are learning: in format states x weights
    # stim is a cell with each stimulus (stim{}['data']) and the probability
    #       transition functions (stim{}['gamma'] and stim{}['xi'])

    # note that this transition function is dependent on where we are
    # transitioning FROM, and relies on each of the other possible states we
    # could be transitioning to; so we cannot minimize these independently.
    # Thus we are really going to find the gradient of all transition filters
    # originating FROM some state

    numstates = stim[0]['numstates']
    numbins = stim[0]['numtotalbins']

    trans_w = np.reshape(trans_w,(numbins,numstates)).T

    allgrad = np.zeros((numstates, numbins));
    allvalue = 0
    totalT = 0

    thisLambda = options['trans_lambda']

    for trial in range(len(stim)):
        totalT = totalT + stim[trial]['data'].shape[1]

    for trial in range(len(stim)):
        tgrad = np.zeros(trans_w.shape[0])
        T = stim[trial]['data'].shape[1]-1
        filtpower = np.sum(np.tile(np.expand_dims(trans_w,axis=2),(1,1,T)) * np.tile(np.reshape(stim[trial]['data'][:,1:],(1,numbins,T)),(numstates,1,1)),axis=1)
        # now filtpower is states x time

        value = -stim[trial]['gamma'][statenum,0:-1] * np.log(1 + np.sum(np.exp(filtpower[np.setdiff1d(np.arange(numstates),statenum),:]),axis=0))
        if stim[trial]['xi'].shape[2] == 1:
            tgrad = stim[trial]['xi'][statenum,:,:].T;
        else:
            tgrad = stim[trial]['xi'][statenum,:,:]

        i=statenum
        offset = stim[trial]['gamma'][statenum,0:-1] / (1 + np.sum(np.exp(filtpower[np.setdiff1d(np.arange(numstates),statenum),:]),axis=0))
        for j in range(numstates):
            if statenum!=j:
                value = value + stim[trial]['xi'][statenum,j,:] *filtpower[j,:]
                tgrad[j,:] = tgrad[j,:] - np.exp(filtpower[j,:])*offset;
            else:
                tgrad[j,:] = 0

        tgrad = np.sum(np.tile(np.reshape(tgrad,(numstates,1,T)),(1,numbins,1)) * np.tile(np.reshape(stim[trial]['data'][:,1:],(1,numbins,T)),(numstates,1,1)),axis=2)

        allgrad = allgrad + tgrad;
        allvalue = allvalue + np.sum(value);


    gradRegularization = np.zeros(allgrad.shape)
    valueRegularization = 0

    # if options.L2smooth
    #     Dx1 = spdiags(ones(size(trans_w,2)-1,1)*[-1 1],0:1,size(trans_w,2)-1-1,size(trans_w,2)-1); 
    #     Dx = Dx1'*Dx1; % computes squared diffs
        
    #     for fstart=(options.numFilterBins+1):options.numFilterBins:(size(trans_w,2)-1)
    #         Dx(fstart,fstart) = 1;
    #         Dx(fstart-1,fstart-1) = 1;
    #         Dx(fstart-1,fstart) = 0;
    #         Dx(fstart,fstart-1) = 0;
    #     end
    #     D = blkdiag(Dx,0);

    #     if options.AR_lambda ~= -1
    #         if length(options.smoothLambda) == 1
    #             options.smoothLambda = repmat(options.smoothLambda(1),[size(trans_w,1)-1,size(trans_w,2)]);
    #             options.smoothLambda(:,options.ARvec) = options.AR_lambda;
    #         end

    #         gradRegularization(setdiff(1:numstates,statenum),:) = gradRegularization(setdiff(1:numstates,statenum),:) + options.smoothLambda .* (D * trans_w(setdiff(1:numstates,statenum),:)')';
    #         valueRegularization = valueRegularization + sum(sum(((options.smoothLambda/2) .* (D * trans_w(setdiff(1:numstates,statenum),:)')').^2));
    #     else
    #         gradRegularization(setdiff(1:numstates,statenum),:) = gradRegularization(setdiff(1:numstates,statenum),:) + options.smoothLambda .* (D * trans_w(setdiff(1:numstates,statenum),:)')';
    #         valueRegularization = valueRegularization +  sum(sum(((options.smoothLambda/2) .* (D * trans_w(:,:)')').^2));
    #     end
    # end
    # if thisLambda ~= 0
    #     if options.AR_lambda ~= -1
    #         gradRegularization = gradRegularization + [thisLambda * trans_w(:,options.stimvec), options.AR_lambda * trans_w(:,options.ARvec)];
    #         valueRegularization = valueRegularization + (thisLambda/2) * sum(sum(trans_w(:,options.stimvec).^2)) + (options.AR_lambda/2) * sum(sum(trans_w(:,options.ARvec).^2));
    #     else
    #         gradRegularization = gradRegularization + thisLambda * trans_w(:,:);
    #         valueRegularization = valueRegularization + (thisLambda/2) * sum(sum(trans_w(:,:).^2));
    #     end
    # end
    print(allvalue)

    # allgrad = -allgrad/totalT + gradRegularization;
    # allvalue = -allvalue/totalT + valueRegularization;

    # WHY IS THIS HAPPENING??
    allgrad = allgrad/totalT + gradRegularization;
    allvalue = -allvalue/totalT + valueRegularization;

    # print(allgrad.shape)
    # print(allgrad)
    # plt.plot(allgrad.T)
    # plt.show()

    if allvalue < 0:
        print('why oh why oh why');
        exit()


    return allvalue, np.ndarray.flatten(allgrad)

