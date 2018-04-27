import numpy as np
from scipy.optimize import minimize
from scipy.io import savemat, loadmat

import scipy.stats
import scipy.ndimage.filters

from HMMGLMLikelihoods import *
from HMMGLMLearningFun import *

def getDefaultOptions(options):
    if 'L2Smooth' not in options:
        options['L2smooth'] = 0
    if 'smoothLambda' not in options:
        options['smoothLambda'] = 0
    if 'numFilterBins' not in options:
        options['numFilterBins'] = 30
    if 'trans_lambda' not in options:
        options['trans_lambda'] = 0
    if 'emit_lambda' not in options:
        options['emit_lambda'] = 0
    if 'GLMemissions' not in options:
        options['GLMemissions'] = True
    if 'GLMtransitions' not in options:
        options['GLMtransitions'] = True

    return options


def GLMHMMtrain(symb, emit_w, trans_w, stim, analog_emit_w, analog_symb, outfilename, options):
    ###################################################
    # First set everything up
    ###################################################

    # Set up options (reduced set compared to Matlab version)
    options = getDefaultOptions(options)
    if len(symb) > 0:
        options['symbExists'] = True
    else:
        options['symbExists'] = False


    totalTrials = np.max((len(symb),len(analog_symb)))  # how many different trials are we fitting?
    # figure out from the data how many symbols there are to fit (though, really, this should be the same as the second
    # dimension of emit_w + 1 so I'm not sure why I'm doing it this way...)
    allsymb = []
    for i in range(len(symb)):
        allsymb.append(np.unique(symb[i]))

    allsymb = np.unique(allsymb)
    numsymb = len(allsymb)

    if 'maxiter' not in options:
        maxiter = 1000;
    else:
        maxiter = options['maxiter'];

    loglik = np.zeros(maxiter+1)
    loglik[0] = -10000000;
    thresh = 1e-4;

    if type(emit_w) != list:
        numstates = emit_w.shape[0]
    elif type(analog_emit_w) != list:
        numstates = analog_emit_w.shape[0]

    numemissions = emit_w.shape[1]
    numtotalbins = np.max((emit_w.shape[2],trans_w.shape[2]))

    if 'analog_flag' in options:
        numAnalogParams = size(analog_symb[0],1)
        numAnalogEmit = zeros(numAnalogParams,1)
    else:
        numAnalogParams = 0;

    prior = []
    gamma = []
    xi = []
    for trial in range(totalTrials):
        prior.append(np.ones(numstates)/numstates)
        gamma.append(np.ones((numstates,symb[trial].shape[0])))
        xi.append(np.zeros(0))
        gamma[trial] = gamma[trial] / np.tile(np.sum(gamma[trial],axis=0),(numstates,1))

    # xi[np.max(len(symb),len(analog_symb))] = [];


    ###################################################
    # Then the E-step
    ###################################################

    # First we need to know the likelihood of seeing each symbol given the filters
    # as well as the likelihood of seeing a transition from step to step
    symblik = []
    transition = []
    for trial in range(np.max((len(symb),len(analog_symb)))):
        if options['symbExists']:
            symblik.append(GLMHMM_SymbLik(emit_w,stim[trial],symb[trial]))

        transition.append(GLMHMM_TransLik(trans_w,stim[trial]))

    outvars = []
    for ind in range(maxiter):
        print('fitting iteration ' + str(ind))

        for trial in range(totalTrials):
            if options['symbExists']:
                emitLikeli = symblik[trial]

            # things get funky if the likelihood is exactly 0
            emitLikeli[emitLikeli < np.finfo(float).eps*1e3] = np.finfo(float).eps*1e3    # do we need this?

            # use the forward backward algorithm to estimate the probability of being in 
            # a given state (gamma), probability of transitioning between states (xi)
            # and hold the prior for initialization of next round of fitting
            gamma[trial],xi[trial],prior[trial] = computeTrialExpectation(prior[trial],emitLikeli,transition[trial])


        ###################################################
        # Now the M-step
        ###################################################

        # First do gradient descent for the emission filter
        # note: 'symbExists' option is holdover from the option to fit only analog features (currently only in matlab)
        if options['symbExists']:
            print('fitting categorical emission filters')

            # clear newstim;
            # format the data appropriately to pass into the minimization function
            # (note: I can probably do this more cleanly)
            newstim = []
            for trial in range(totalTrials):
                # please don't ask me why I decided it was a good idea to call the number of emissions 'numstates' here. Just roll with it
                newstim.append({'xi':xi[trial],'gamma':gamma[trial],'numstates':numemissions,'emit':symb[trial]})
                if options['GLMemissions']:
                    newstim[trial]['data'] = stim[trial]
                    newstim[trial]['numtotalbins'] = numtotalbins;
                else:
                    newstim[trial]['data'] = stim[trial][-1,:]
                    newstim[trial]['numtotalbins'] = 1

            for i in range(numstates):
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                # I am using the scipy minimization function and passing the analytic value and the gradient to it
                # NOTE that I also COULD compute the Hessian but IME it ends up not speeding the fitting up much because
                # it is very slow and memory intensive to compute on each iteration
                outweights = minimize(lambda x: emitLearningFun(x, newstim, i, options), np.reshape(emit_w[i,:,:].T,(emit_w.shape[1]*emit_w.shape[2],1)), jac=True, method='BFGS');
                emit_w[i,:,:] = np.reshape(outweights['x'],(emit_w.shape[2],emit_w.shape[1])).T    # make sure this is reformatted properly!!!
                # NOTE: this returns the inverse Hessian so we can get the error bars from that if we want to


        # gradient descent for the transition filter
        print('fitting transition filters')

        # this is essentially the same as fitting the emission filter
        newstim = []
        for trial in range(len(stim)):
            newstim.append({'xi':xi[trial],'gamma':gamma[trial],'numstates':numstates,'emit':symb[trial]})
            if options['GLMemissions']:
                newstim[trial]['data'] = stim[trial]
                newstim[trial]['numtotalbins'] = numtotalbins;
            else:
                newstim[trial]['data'] = stim[trial][-1,:]
                newstim[trial]['numtotalbins'] = 1

        for i in range(numstates):
            # print('in state ' + str(i))
            outweights = minimize(lambda x: transLearningFun(x, newstim, i, options), np.reshape(trans_w[i,:,:].T,(trans_w.shape[2]*trans_w.shape[1],1)), jac=True, method='BFGS')
            trans_w[i,:,:] = np.reshape(outweights['x'],(trans_w.shape[2],trans_w.shape[1])).T


        # Now we have done the E and the M steps! Just save the likelihoods
        analogemit_lik = 0;
        trans_lik = 0;
        emit_lik = 0;
        for trial in range(np.max((len(symb),len(analog_symb)))):
            if options['symbExists']:
                symblik[trial] = GLMHMM_SymbLik(emit_w,stim[trial],symb[trial])

            transition[trial] = GLMHMM_TransLik(trans_w,stim[trial])

            emit_lik = emit_lik + -np.sum(gamma[trial] * np.log(symblik[trial]));
            trans_lik = trans_lik + -np.sum(xi[trial] * np.log(transition[trial][:,:,1:]))

        # log likelihood: sum(gamma(n) * log) + tgd_lik + pgd_lik
        basic_lik = np.zeros(totalTrials)
        for i in range(len(symb)):
            gamma[i][gamma[i][:,0] == 0,0] = np.finfo(float).eps
            basic_lik[i] = -np.sum(gamma[i][:,0] * np.log(gamma[i][:,0]))

        loglik[ind+1] = np.sum(basic_lik) + emit_lik + trans_lik + analogemit_lik

        outvars.append({'emit_w':emit_w,'trans_w':trans_w})
        if options['symbExists']:
            outvars[ind]['emit_lik'] = emit_lik;

        outvars[ind]['trans_lik'] = trans_lik

        outvars[ind]['loglik'] = loglik[1:ind+1];
        outvars[ind]['gamma'] = gamma;
        outvars[ind]['trans_lambda'] = options['trans_lambda']
        outvars[ind]['emit_lambda'] = options['emit_lambda']
        outvars[ind]['smoothLambda'] = options['smoothLambda']

        savemat(outfilename,{'outvars':outvars,'options':options})

        print('log likelihood: ' + str(loglik[ind+1]))

        # I have been stopping if the % change in log likelihood is below some threshold
        if (abs(loglik[ind+1]) - abs(loglik[ind]))/abs(loglik[ind]) < thresh:
            break

    print('FINISHED')
    # return outvars


def makeStim(totaltime, bins, numinputs,stimscale):
    stim = np.zeros((bins,totaltime+bins-1,numinputs))
    stim[0,:,:] = scipy.ndimage.filters.gaussian_filter(np.random.randn(totaltime+bins-1,numinputs),stimscale)
    for i in range(1,bins):
        stim[i,0:totaltime,:] = stim[0,i:(totaltime+i),:]

    return stim[:,0:totaltime,:]

def makePeakedSampleFilter(tau,bins):
    return scipy.stats.gamma.pdf(np.linspace(0,bins),a=tau)[0:bins]

def exampleOne():
    tau = 4
    bins = 30
    noiseSD = 0.1
    numinputs = 3
    totaltime = 10000
    stimscale = 1
    numsamples = 5
    numstates = 2
    numRealStates = 2

    stim = []
    states = []
    outputTrace = []
    outputStim = []
    for ns in range(numsamples):
        output = np.zeros((numRealStates,totaltime))
        stim.append(makeStim(totaltime, bins, numinputs,stimscale) + np.random.randn(bins,totaltime,numinputs)*noiseSD)
        filt = makePeakedSampleFilter(tau, bins);

        p1 = np.exp(np.matmul(stim[ns][:,:,0].T,filt.T) + np.matmul(stim[ns][:,:,1].T,-filt.T))
        output[0,:] = p1/(1+p1) > 0.5
        p1 = np.exp(np.matmul(stim[ns][:,:,0].T,-filt.T) + np.matmul(stim[ns][:,:,1].T,filt.T))
        output[1,:]= p1/(1+p1) > 0.5

        p1 = np.exp(np.matmul(stim[ns][:,:,-1].T,filt.T))
        states.append(p1/(1+p1) > 0.5)

        outputTrace.append(np.zeros(totaltime))
        for ss in range(numRealStates):
            outputTrace[ns][states[ns] == ss] = output[ss,states[ns] == ss]

        finalStim = np.append(stim[ns][:,:,0],stim[ns][:,:,1],axis=0)
        finalStim = np.append(finalStim,stim[ns][:,:,2],axis=0)
        finalStim = np.append(finalStim,np.ones((1,totaltime)),axis=0)
        outputStim.append(finalStim)

    model_options = {'emit_lambda':0,'trans_lambda':0,'L2smooth':0,'smoothLambda':0,'numstates':2,'numFilterBins':bins}
    emit_w = np.zeros((numstates,1,bins*numinputs + 1))     # states x emissions-1 x filter bins
    trans_w = np.zeros((numstates,numstates,bins*numinputs + 1))     # states x states x filter bins [diagonals are ignored]

    for ss in range(numstates):
        for ee in range(numinputs):
            emit_w[ss,0,(ee-1)*bins + np.arange(bins)] = np.exp(-np.arange(bins)/30) * np.round(np.random.random(1)*2 - 1)

            for ss2 in range(numstates):
                trans_w[ss,ss2,(ee-1)*bins + np.arange(bins)] = np.exp(-np.arange(bins)/30) * np.round(np.random.random(1)*2 - 1)

    return outputTrace, outputStim, emit_w, trans_w, model_options

if __name__ == '__main__':
    # some example code to run here
    # outputTrace, outputStim, emit_w, trans_w, model_options = exampleOne()

    data = loadmat('../tmp.mat')
    outputTrace = data['outputTrace'][0]
    emit_w = data['features']['emit_w'][0][0]
    trans_w = data['features']['trans_w'][0][0]
    outputStim = data['outputStim']

    model_options = {'emit_lambda':0,'trans_lambda':0,'L2smooth':0,'smoothLambda':0,'numstates':2,'numFilterBins':30}
    outputStim = list(outputStim[0,:])
    for ii,trace in enumerate(outputTrace):
        outputTrace[ii] = np.squeeze(outputTrace[ii])

    [outvars] = GLMHMMtrain(outputTrace, emit_w, trans_w, outputStim, [], [], 'out.mat', model_options)

