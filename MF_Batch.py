'''
% Version 1.000
%
% Code provided by Gang Chen, SUNY at Buffalo
% gangchen@buffalo.edu
% 
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.

'''


import os  
import scipy.io as sio  
import numpy as np
import numpy.ma as ma
import theano
import math
from collections import OrderedDict
from theano import tensor as T

floatX = theano.config.floatX

class MF_Batch(object):
    def __init__(self, i, j, num_user, num_item, factors, init_mean, init_stdev, steps=5000, alpha=0.0002, beta=0.02):
    
        # user & item latent vectors
        P_init = np.random.normal(loc=init_mean, scale=init_stdev, size=(num_user, factors))
        Q_init = np.random.normal(loc=init_mean, scale=init_stdev, size=(num_item, factors))
        self.P = theano.shared(value = P_init.astype(floatX), 
                               name = 'U', borrow = True)
        self.Q = theano.shared(value = Q_init.astype(floatX),
                               name = 'V', borrow = True)
    
        self.pred = self.P[i,:].dot(self.Q[j,:].T).diagonal() 
    
        self.params = [self.P, self.Q]
        self.i = i
        self.j = j
        self.params2 = OrderedDict()
        self.params2['P'] = self.P
        self.params2['Q'] = self.Q
    '''
    P_i = T.matrix()
    Q_j = T.matrix()
    i = T.lvector('i')
    j = T.lvector('j')
    x = T.lvector('x')
    pred = P_i.dot(Q_j).diagonal() 
    error =T.sum( T.sqr(pred - x))
    regularization = (beta/2.0) * ((P_i**2).sum() + (Q_j**2).sum())
    cost = error + regularization
    gp, gq = T.grad(cost=cost, wrt=[P_i, Q_j])
    train = theano.function(inputs=[i, j, x],
                          givens=[(P_i, P[i, :]), (Q_j, Q[:, j])],
                          updates=[(P, T.inc_subtensor(P[i, :], -gp * alpha)),
                                   (Q, T.inc_subtensor(Q[:, j], -gq * alpha))])
    '''
    def errors(self, x, beta): 
        error =T.sum( T.sqr(self.pred - x))
        regularization = (beta/2.0) * ((self.P[self.i,:]**2).sum() + (self.Q[self.j,:]**2).sum())
        cost = error + regularization
        return cost, error

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    """Adam updates
    Adam updates implemented as in [1]_.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        Learning rate
    beta1 : float or symbolic scalar
        Exponential decay rate for the first moment estimates.
    beta2 : float or symbolic scalar
        Exponential decay rate for the second moment estimates.
    epsilon : float or symbolic scalar
        Constant for numerical stability.
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.
    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def optimization_adam(trainvec, testvec, n_epochs, batch_size, alpha=0.001, beta=0.1):
    i = T.lvector('i')
    j = T.lvector('j')
    x = T.dvector('x')
    num_user=6040
    num_item=3952
    factors = 20
    init_mean = 0
    init_stdev = 0.02
    mfobj = MF_Batch(i, j, num_user, num_item, factors, init_mean, init_stdev)
    regcost, error = mfobj.errors(x, beta)
    
    #f_grad = theano.function([i, j, x], grads, name='f_grad')
    # lr = theano.shared(alpha*np.ones(1, dtype='float32'))
    #lr = T.scalar(name='lr')
    updates = Adam(regcost, mfobj.params, alpha)
    train_model = theano.function(inputs=[i, j, x], 
                                  updates=updates, 
                                  outputs=error)  
    test_model = theano.function(inputs=[i, j, x], 
                                  outputs=error)   
    
    
    mean_rating = np.mean(trainvec[:, 2])
    done_looping = False
    epoch = 0
    N = len(trainvec)
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        totalErrors = 0
        testErrors = 0
        for k in range(int(math.floor(N / batch_size))):
            batch = np.arange(k * batch_size, min(N-1, (k + 1) * batch_size))
            idi = trainvec[batch, 0]-1
            idj = trainvec[batch, 1]-1
            ratings = trainvec[batch, 2] -  mean_rating
            batch_cost = train_model(idi, idj, ratings)
            totalErrors += batch_cost
            
        NN = len(testvec)   
        batch_size =1000
        for k in range(int(math.floor(NN / batch_size))):
            batch = np.arange(k * batch_size, min(NN-1, (k + 1) * batch_size))
            p_idx = testvec[batch, 0]-1
            q_idx = testvec[batch, 1]-1
            ratings = testvec[batch, 2] - mean_rating
            testErrors += test_model(p_idx, q_idx, ratings)
        print("the training cost at epoch {} is {}, and the testing error is {}".format(epoch, np.sqrt(totalErrors/N), np.sqrt(testErrors/NN)))
    
        # test it on the test dataset
    NN = len(testvec)   
    batch_size =1000
    diff = 0
    for k in range(int(math.floor(NN / batch_size))):
        batch = np.arange(k * batch_size, min(NN-1, (k + 1) * batch_size))
        p_idx = testvec[batch, 0]-1
        q_idx = testvec[batch, 1]-1
        ratings = testvec[batch, 2] - mean_rating
        diff += test_model(p_idx, q_idx, ratings)
        
    print("Total average test error for {} instances is {}".format(NN, np.sqrt(diff/NN))) 
    
def adadelta(lr, tparams, grads, i, j, x, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    i, j Theano variable
        Model input index
    x: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize
    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    zipped_grads = [theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX),
                                  name='%s_grad' % k) for k, p in tparams.items()]
    running_up2 = [theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX),
                                 name='%s_rup2' % k) for k, p in tparams.items()]
    running_grads2 = [theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([i, j, x], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

'''       
def get_batch(self, data, batch_size):
    ratings, idi, idj = [], [], []
    for i in range(batch_size):
        # sample a user
        u = np.random.randint(0, self.num_user)
        # sample a positive item
        i = self.train[u][np.random.randint(0, len(self.train[u]))][0]
            # sample a negative item
        j = np.random.randint(0, self.num_item)
        while j in self.items_of_user[u]:
            j = np.random.randint(0, self.num_item)
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)
        return (users, pos_items, neg_items) 
'''
def optimization_adadelta(trainvec, testvec, n_epochs, batch_size, alpha=0.001, beta=0.1):
    i = T.lvector('i')
    j = T.lvector('j')
    x = T.dvector('x')
    num_user=6040
    num_item=3952
    factors = 20
    init_mean = 0
    init_stdev = 0.02
    mfobj = MF_Batch(i, j, num_user, num_item, factors, init_mean, init_stdev)
    regcost, error = mfobj.errors(x, beta)
    grads = T.grad(cost=regcost, wrt=[mfobj.P, mfobj.Q])
    #f_grad = theano.function([i, j, x], grads, name='f_grad')
    lr = T.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, mfobj.params2, grads, i, j, x, regcost)
    
    test_model = theano.function(inputs=[i, j, x], 
                                  #givens=[(mfobj.P[i, :]), mfobj.Q[:, j]], 
                                  outputs=error)   
    
    
    mean_rating = np.mean(trainvec[:, 2])
    done_looping = False
    epoch = 0
    N = len(trainvec)
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        totalErrors = 0
        testErrors = 0
        for k in range(int(math.floor(N / batch_size))):
            batch = np.arange(k * batch_size, min(N-1, (k + 1) * batch_size))
            idi = trainvec[batch, 0]-1
            idj = trainvec[batch, 1]-1
            ratings = trainvec[batch, 2] -  mean_rating
            batch_cost = f_grad_shared(idi, idj, ratings)
            f_update(alpha)
            totalErrors += batch_cost
            
        NN = len(testvec)   
        batch_size =1000
        for k in range(int(math.floor(NN / batch_size))):
            batch = np.arange(k * batch_size, min(NN-1, (k + 1) * batch_size))
            p_idx = testvec[batch, 0]-1
            q_idx = testvec[batch, 1]-1
            ratings = testvec[batch, 2] - mean_rating
            testErrors += test_model(p_idx, q_idx, ratings)
        print("the training cost at epoch {} is {}, and the testing error is {}".format(epoch, np.sqrt(totalErrors/N), np.sqrt(testErrors/NN)))
    
        # test it on the test dataset
    NN = len(testvec)   
    batch_size =1000
    diff = 0
    for k in range(int(math.floor(NN / batch_size))):
        batch = np.arange(k * batch_size, min(NN-1, (k + 1) * batch_size))
        p_idx = testvec[batch, 0]-1
        q_idx = testvec[batch, 1]-1
        ratings = testvec[batch, 2] - mean_rating
        diff += test_model(p_idx, q_idx, ratings)
        
    print("Total average test error for {} instances is {}".format(NN, np.sqrt(diff/NN))) 

def optimization_sgd(trainvec, testvec, n_epochs, batch_size, alpha=0.01, beta=0.05):
    i = T.lvector('i')
    j = T.lvector('j')
    x = T.dvector('x')
    num_user=6040
    num_item=3952
    factors = 20
    init_mean = 0
    init_stdev = 0.02
    mfobj = MF_Batch(i, j, num_user, num_item, factors, init_mean, init_stdev)
    regcost, error = mfobj.errors(x, beta)
    gp, gq = T.grad(cost=regcost, wrt=[mfobj.P, mfobj.Q])
    updates=[(mfobj.P, T.inc_subtensor(mfobj.P[i, :], - gp[i,:] * alpha)),
                                   (mfobj.Q, T.inc_subtensor(mfobj.Q[j,:], -gq[j,:] * alpha))]
    train_model = theano.function(inputs=[i, j, x], 
                                  #givens=[(mfobj.P[i, :]), mfobj.Q[:, j]], 
                                  outputs=regcost,
                                  updates=updates)
    
    test_model = theano.function(inputs=[i, j, x], 
                                  #givens=[(mfobj.P[i, :]), mfobj.Q[:, j]], 
                                  outputs=error)   
    
    
    mean_rating = np.mean(trainvec[:, 2])
    done_looping = False
    epoch = 0
    N = len(trainvec)
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        totalErrors = 0
        testErrors = 0
        for k in range(int(math.floor(N / batch_size))):
            batch = np.arange(k * batch_size, min(N-1, (k + 1) * batch_size))
            idi = trainvec[batch, 0]-1
            idj = trainvec[batch, 1]-1
            ratings = trainvec[batch, 2] -  mean_rating
            minibatch_cost = train_model(idi, idj, ratings)
            totalErrors += minibatch_cost
            
        NN = len(testvec)   
        batch_size =1000
        for k in range(int(math.floor(NN / batch_size))):
            batch = np.arange(k * batch_size, min(NN-1, (k + 1) * batch_size))
            p_idx = testvec[batch, 0]-1
            q_idx = testvec[batch, 1]-1
            ratings = testvec[batch, 2] - mean_rating
            testErrors += test_model(p_idx, q_idx, ratings)
        print("the training cost at epoch {} is {}, and the testing error is {}".format(epoch, np.sqrt(totalErrors/N), np.sqrt(testErrors/NN)))
    
        # test it on the test dataset
    NN = len(testvec)   
    batch_size =1000
    diff = 0
    for k in range(int(math.floor(NN / batch_size))):
        batch = np.arange(k * batch_size, min(NN-1, (k + 1) * batch_size))
        p_idx = testvec[batch, 0]-1
        q_idx = testvec[batch, 1]-1
        ratings = testvec[batch, 2] - mean_rating
        diff += test_model(p_idx, q_idx, ratings)
        
    print("Total average test error for {} instances is {}".format(NN, np.sqrt(diff/NN)))    
    
if __name__ == "__main__":  
    filepath = './moviedata.mat'
    mat_contents = sio.loadmat(filepath)
    trainvec = mat_contents['train_vec']
    testvec = mat_contents['probe_vec']
    #optimization_sgd(trainvec, testvec, 100, 500)
    '''
    params2 = OrderedDict()
    params2['P'] = 0.01 * np.random.rand(20, 30)
    params2['Q'] = 0.01 * np.random.rand(20, 30)
    tparams = init_tparams(params2)
    for k, p in tparams.items():
        print('%s_grad' % k)
        name='{}_grad'.format(k)
        v = p.get_value() * 0.0
        zipped_grads = [theano.shared(p.get_value() * 0.0,
                                  name='{}_grad'.format(k))]

    '''
    
     # use True or False to decide use which optimization approach
    adam = False 
    sgd = True
    ada = False
    if ada: 
        optimization_adadelta(trainvec, testvec, 100, 500)
    if adam:
        optimization_adam(trainvec, testvec, 100, 500)
    
    if sgd: 
        optimization_sgd(trainvec, testvec, 100, 500)
    '''
    # trainvec = trainvec[1:10001, :]
    mean_rating = np.mean(trainvec[:, 2])
    # Rating matrix
    nummovies = 3952;  # Number of movies 
    numusers = 6040; 
    num_feat = 20;
    
    
    currentdir = os.path.dirname(os.path.realpath(__file__))
    # save file to speed up
    tmpfile = '/tmpfile25.bin'
    # the number of training data
    numdata = len(trainvec)
    NUM = 25 # the number of correlation
    if not os.path.isfile(currentdir+tmpfile):
        # compute the rating matrix
        R = np.zeros((numusers, nummovies))
        for i in range(len(trainvec)):
            # trainvec[i,0] = trainvec[i,0] -1 # start from 0
            # trainvec[i,1] = trainvec[i,1] -1 # start from 0
            R[trainvec[i,0]-1, trainvec[i,1]-1] = trainvec[i,2] 
            

    K = 20
    P = np.random.rand(numusers, K)
    Q = np.random.rand(K, nummovies)        
    #P_theano_sgd, Q_theano_sgd = matrix_factorization_sgd(ma.masked_array(R, mask=R==0), P, Q)      
        
    '''  