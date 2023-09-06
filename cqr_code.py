import sys
import torch
import numpy as np
import torch_models
from functools import partial
# from cqr import tune_params_cv
# from nonconformist.cp import IcpRegressor
# from nonconformist.base import RegressorAdapter
# from skgarden import RandomForestQuantileRegressor

class AllQNet_RegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 quantiles=[.05, .95],
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0,
                 use_rearrangement=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation
        use_rearrangement : boolean, use the rearrangement algorithm (True)
                            of not (False). See reference [1].

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """
        super(AllQNet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        if use_rearrangement:
            self.all_quantiles = torch.from_numpy(np.linspace(0.01,0.99,99)).float()
        else:
            self.all_quantiles = self.quantiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.all_q_model(quantiles=self.all_quantiles,
                                              in_shape=in_shape,
                                              hidden_size=hidden_size,
                                              dropout=dropout)
        self.loss_func = torch_models.AllQuantileLoss(self.all_quantiles)
        self.learner = torch_models.LearnerOptimizedCrossing(self.model,
                                                             partial(learn_func, lr=lr, weight_decay=wd),
                                                             self.loss_func,
                                                             device=device,
                                                             test_ratio=self.test_ratio,
                                                             random_state=self.random_state,
                                                             qlow=self.quantiles[0],
                                                             qhigh=self.quantiles[1],
                                                             use_rearrangement=use_rearrangement)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, self.batch_size)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        return self.learner.predict(x)