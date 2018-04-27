
import numpy as np
import copy
import scipy.misc

def GLMHMM_SymbLik(emit_w,stim,symb):
	# emit_w is in shape [states,symbols,time]
	# stim is in shape [history,time]
    numstates = emit_w.shape[0]
    numsymb = emit_w.shape[1]

    # put the stimulus in a different format for easier multiplication
    stim_e = np.tile(np.reshape(stim,(1,1,stim.shape[0], stim.shape[1])),(numstates,numsymb,1,1))
    symblik = np.zeros((emit_w.shape[0],len(symb)))

    # likelihood is exp(k*w) / (1 + sum(exp(k*w)))
    for t in range(symb.shape[0]):
        symblik[:,t] = 1 / (1 + np.sum(np.exp(np.sum(emit_w*stim_e[:,:,:,t],axis=2)),axis=1))

        # if the emission symbol is 0, we have 1 on the numerator otherwise exp(k*w)
        if symb[t] != 0:
            symblik[:,t] = symblik[:,t] * np.exp(np.sum(emit_w[:,int(symb[t])-1,:]*stim_e[:,int(symb[t])-1,:,t],axis=1))

        if np.any(np.isnan(symblik[:,t])):
            print('oh dear')

    return symblik

def GLMHMM_TransLik(trans_w,stim):
    # compute transition likelihood
    # trans_w is in shape [states,states,time]
    # stim is in shape [history,time]

    T = stim.shape[1]
    numstates = trans_w.shape[0]
    numbins = trans_w.shape[2]
    transition = np.zeros((numstates,numstates,T))
    for i in xrange(numstates):
        filtpower = np.sum(np.tile(np.reshape(trans_w[i,:,:],(numstates,numbins,1)),(1,1,T)) * np.tile(np.reshape(stim,(1,stim.shape[0],T)),(numstates,1,1)),axis=1)

        # there is no filter for going from state i to state i
        filtpower[i,:] = 0
        for j in xrange(numstates):
            transition[i,j,:] = np.exp(filtpower[j,:] - scipy.misc.logsumexp(filtpower,axis=0))

    return transition

def computeTrialExpectation(prior,likeli,transition):
    # Forward-backward algorithm, see Rabiner for implementation details
	# http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf

    t = 0
    totalTime = likeli.shape[1]
    numstates = likeli.shape[0]

    # E-step
    # alpha is the forward probability of seeing the sequence
    alpha = np.zeros((prior.shape[0],totalTime))
    scale_a = np.ones(totalTime);

    alpha[:,0] = prior * likeli[:,0]
    alpha[:,t] = alpha[:,t] / np.sum(alpha[:,t]);

    for t in xrange(1,totalTime):
        alpha[:,t] = np.matmul(transition[:,:,t],alpha[:,t-1])
        alpha[:,t] = alpha[:,t] * likeli[:,t]

        # use this scaling component to try to prevent underflow errors
        scale_a[t] = np.sum(alpha[:,t])
        alpha[:,t] = alpha[:,t] / scale_a[t]


    # beta is the backward probability of seeing the sequence
    beta = np.zeros((len(prior),totalTime)) 		# beta(i,t)  = Pr(O(t+1:totalTime) | X(t)=i)
    beta[:,-1] = np.ones(prior.shape[0])/len(prior)

    scale_b = np.ones(totalTime);
    for t in xrange(totalTime-2,0,-1):
        beta[:,t] = np.matmul(transition[:,:,t+1], (beta[:,t+1] * likeli[:,t+1]))
        scale_b[t] = np.sum(beta[:,t])
        beta[:,t] = beta[:,t] / scale_b[t]

    # if any of the values are 0 it's defacto an underflow error so set it to eps
    alpha[alpha == 0] = np.finfo(float).eps
    beta[beta == 0] = np.finfo(float).eps

    # gamma is the probability of seeing the sequence, found by combining alpha and beta
    gamma = np.exp(np.log(alpha) + np.log(beta) - np.tile(np.log(np.cumsum(scale_a)).T,(numstates,1)) - np.tile(np.log(np.flip(np.cumsum(np.flip(scale_b,0)),0)).T,(numstates,1)))
    gamma[gamma == 0] = np.finfo(float).eps
    gamma = gamma / np.tile(np.sum(gamma,axis=0),(numstates,1))

    # xi is the probability of seeing each TRANSITION in the sequence
    xi =np.zeros((len(prior),len(prior),totalTime-1))
    transition2 = copy.copy(transition[:,:,1:])

    for s1 in xrange(numstates):
        for s2 in xrange(numstates):
            xi[s1,s2,:] = np.log(likeli[s2,1:]) + np.log(alpha[s1,0:-1]) + np.log(np.squeeze(transition2[s1,s2,:]).T) + np.log(beta[s2,1:]) - np.log(np.cumsum(scale_a[:-1])).T - np.log(np.flip(np.cumsum(np.flip(scale_b[1:],0)),0)).T
            xi[s1,s2,:] = np.exp(xi[s1,s2,:])

    xi[xi == 0] = np.finfo(float).eps
    # renormalize to make sure everything adds up properly
    xi = xi / np.tile(np.sum(np.sum(xi,axis=0),axis=0),(numstates,numstates,1))

    # save the prior initialization state for next time
    prior = gamma[:,0]

    return gamma,xi,prior
