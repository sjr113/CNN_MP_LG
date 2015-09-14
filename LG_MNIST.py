# -*- coding: utf-8 -*-
# coding=utf-8
__author__ = 'shen'

import cPickle
import os
import sys
import numpy as np
import gzip
import timeit


class LogisticRegression(object):
    def __init__(self, n_in, n_out):
        # :param input: symbolic variable that describes the input of the
        # architecture (one minibatch)
        # :param n_in: number of input units, the dimension of the space in
        # which the data points lie
        # :param n_out: number of output units, the dimension of the space in
        # which the labels lie
        # input 为输入样本，大小为m*n m为样本个数 n=n_in
        # n_in表示输入样本x的特征的维数
        # n_out表示样本的类别数
        self.W = np.zeros([n_in, n_out])
        self.b = np.zeros([n_out, ])

        # keep track of model input
        self.input = []

        # parameters of the model
        self.params = [self.W, self.b]

        self.p_y_given_x = np.zeros([(np.shape(self.input))[0], n_out])
        self.y_pred = np.zeros([(np.shape(self.input))[0], ])

        self.learning_rate = 0.13
        self.y = []

        self.weight_lambda = 1e-6

    def compute_p_y_given_x(self):
        # self.update_grad()
        # print "------------------------"
        # print np.shape(self.input)
        # print np.shape(self.W)
        # print np.shape(self.b)
        # print "wx_b"
        wx_b = np.exp(np.dot(self.input, self.W) + self.b)

        # print wx_b
        # self.p_y_given_x表示将样本x分类为类别y的概率 P(y|x,theta)
        self.p_y_given_x = np.true_divide(wx_b, np.sum(wx_b, axis=1).reshape([np.shape(wx_b)[0], 1]))
        # print "self.p_y_given_x"
        # print self.p_y_given_x

    def compute_y_pred(self):
        self.compute_p_y_given_x()
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # 选取概率值最大的类别作为最后的分类结果
        self.y_pred = np.argmax(self.p_y_given_x, axis=1)
        # print "y_pred"
        # print self.y_pred

    def negative_log_likelihood(self):
        # 计算loss function的值
        # Return the mean of the negative log-likelihood of the prediction
        # of this model under a given target distribution.

        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # :param y: corresponds to a vector that gives for each example the
        # correct label

        # Note: we use the mean instead of the sum so that
        # the learning rate is less dependent on the batch size

        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # y为各个样本对应的正确的label

        self.compute_p_y_given_x()
        return -np.mean(np.log(self.p_y_given_x)[np.arange(self.y.shape[0]), self.y])
        # end-snippet-2

    def compute_grad(self):
        # Y_hat = np.exp(np.dot(classifier, data_x))
        # prob = Y_hat / np.sum(Y_hat, axis = 0)#probabilities
        # C x N array, element(i,j)=1 if y[j]=i
        self.compute_p_y_given_x()
        ground_truth = np.zeros_like(self.p_y_given_x)
        ground_truth[tuple([range(len(self.y)), self.y])] = 1.0
        # print "ground_truth"
        # print ground_truth
        input2 = np.ones([np.shape(self.input)[0], 1])
        data = np.append(self.input, input2, axis=1)
        # print "data"
        # print data
        # loss = -np.sum(ground_truth*np.log(prob)) / num_data + 0.5*reg*np.sum(W*W)
        # loss = -np.sum(ground_truth*np.log(self.p_y_given_x)) / (np.shape(self.input))[0]
        d_params = (-np.dot(data.transpose(), ground_truth - self.p_y_given_x))/(np.shape(data))[0]
        # print "d_params"
        # print d_params
        g_W = d_params[:-1,:]
        g_b = d_params[-1, :]
        """
        print "g_W"
        print g_W
        print "g_b"
        print g_b
        """
        return g_W, g_b


    def update_grad(self):
        """
        print "self.input"
        print self.input
        print "self.y"
        print self.y
        print np.shape(self.y)
        print "self.W"
        print self.W
        print np.shape(self.W)
        print "self.b"
        print self.b
        print np.shape(self.b)
        """
        g_W, g_b = self.compute_grad()
        self.W -= self.learning_rate * g_W
        self.b -= self.learning_rate * g_b
        """
        print "update grad"
        print "W"
        print self.W
        print "b"
        print self.b
        """

    def errors(self):
        # Return a float representing the number of errors in the minibatch
        # over the total number of examples of the minibatch ; zero one
        # loss over the size of the minibatch

        # :param y: corresponds to a vector that gives for each example the
        # correct label
        # y是

        # check if y has same dimension of y_pred
        self.compute_y_pred()
        self.y_pred.resize(np.shape(self.y))
        self.y = np.array(self.y)
        if self.y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', self.y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        # if self.y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1

        # else:
            # raise NotImplementedError()
        return np.mean(np.not_equal(self.y_pred, self.y))

    def line_search_alpha(self):  # line search 和update_grad作用是一样的，只是算法不同
        c = 0.5  # 初始化参数C
        tau = 0.5  # 给定参数tau
        g_W, g_b = self.compute_grad()
        slope = (g_W ** 2).sum(axis=0)  # 求g_W的平方
        i = 0
        while i < np.shape(g_b)[0]:
            t_learning_rate = 1  # 初始化一个最大的alpha
            ori_loss = self.negative_log_likelihood()  # 计算当前的loss函数值
            self.W[:, i] -= t_learning_rate * g_W[:, i]  # 更新self.W 的值
            self.b[i] -= t_learning_rate * g_b[i]  # 更新self.W 的值
            pre_learning_rate = t_learning_rate
            while 1:
                tt = c * t_learning_rate * slope[i]
                # self.compute_p_y_given_x()
                cur_loss = self.negative_log_likelihood()  # 计算当前的loss function的值，因为已经改变了self.W

                if cur_loss <= ori_loss - tt:
                    break
                else:
                    t_learning_rate *= tau  # 若不满足line search的终止条件，则更新learning_rate
                    if t_learning_rate < self.learning_rate:
                        self.W[:, 1] += (pre_learning_rate - self.learning_rate) * g_W[:, 1]
                        break
                self.W[:, i] += (pre_learning_rate - t_learning_rate) * g_W[:, i]
                self.b[i] += (pre_learning_rate - t_learning_rate) * g_b[i]  # 更新self.b 的值
                pre_learning_rate = t_learning_rate
            i += 1
        self.learning_rate = t_learning_rate


def load_data(dataset):
    # Loads the dataset

    # :type dataset: string
    # :param dataset: the path to the dataset (here MNIST)
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # which row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input

    rval = [train_set, valid_set, test_set]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    # Demonstrate stochastic gradient descent optimization of a log-linear
    # model

    # This is demonstrated on MNIST.

    # :type learning_rate: float
    # :param learning_rate: learning rate used (factor for the stochastic
    # gradient)

    # :type n_epochs: int
    # :param n_epochs: maximal number of epochs to run the optimizer

    # :type dataset: string
    # :param dataset: the path of the MNIST dataset file from
    # http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print "train"
    print np.shape(train_set_x)
    print np.shape(train_set_y)
    print "valid"
    print np.shape(valid_set_x)
    print np.shape(valid_set_y)
    print "test"
    print np.shape(test_set_x)
    print np.shape(test_set_y)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    # index = T.lscalar()  # index to a [mini]batch
    # index = 0

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    # x = np.matrix('x')  # data, presented as rasterized images
    # "-----------------------------------------------error"
    # y = np.array('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(n_in=28 * 28, n_out=10)
    classifier_validation = LogisticRegression(n_in=28 * 28, n_out=10)
    classifier_test = LogisticRegression(n_in=28 * 28, n_out=10)
    # classifier = LogisticRegression(n_in=10, n_out=3)
    """
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    # cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    # test_model = Model(classifier, index, y, test_set_x, test_set_y, batch_size)
    # validate_model = Model(classifier, index, y, valid_set_x, valid_set_y, batch_size)

    # "----------------------------------------compute gradience"
    # compute the gradient of cost with respect to theta = (W,b)
    # g_W = T.grad(cost=cost, wrt=classifier.W)
    # g_b = T.grad(cost=cost, wrt=classifier.b)
    # g_W, g_b = compute_grad(p_y_given_x, data_x, y)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    # updates = [(classifier.W, classifier.W - learning_rate * g_W),
    #    (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates
    # train_model = TrainModel(cost, updates, index, train_set_x, train_set_y, batch_size)
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    """
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # print "validation_frequency"
    # print validation_frequency
    # go through this many minibatche before checking the network on the validation set;
    # in this case we check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    valid_error = []
    test_error = []
    epoch_num = []
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            # print "*********************"
            # # print "minibatch_index"
            # print minibatch_index

            # train_model = TrainModel(cost, updates, index, train_set_x, train_set_y, batch_size)
            train_subset_x = train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            train_subset_y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            # print minibatch_index * batch_size, (minibatch_index + 1) * batch_size
            # print train_subset_x
            # print train_subset_y
            # print "---------------------"
            classifier.input = train_subset_x
            classifier.y = train_subset_y
            """
            print "classifier.input"
            print classifier.input
            print "classifier.y"
            print classifier.y
            """
            # minibatch_avg_cost = train_model(minibatch_index)
            classifier.update_grad()
            # classifier.line_search_alpha()
            """
            print "classifier.W"
            print classifier.W
            print "classifier.b"
            print classifier.b
            """
            # print "miao"
            # print classifier.params
            # print np.shape(np.array(classifier.params))
            # print np.shape(classifier.W)
            # print np.shape(classifier.b)

            minibatch_avg_cost = classifier.negative_log_likelihood()
            # print "minibatch_avg_cost"
            # print minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # print "iter"
            # print iter
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = []
                classifier_validation.W = classifier.W
                classifier_validation.b = classifier.b
                """
                print "classifier_validation.W "
                print classifier_validation.W
                print "classifier_validation.b"
                print classifier_validation.b
                """
                for i in xrange(n_valid_batches):
                    valid_subset_x = valid_set_x[i * batch_size: (i + 1) * batch_size]
                    valid_subset_y = valid_set_y[i * batch_size: (i + 1) * batch_size]
                    classifier_validation.input = valid_subset_x
                    # print "classifier_validation.input"
                    # print classifier_validation.input
                    classifier_validation.y = valid_subset_y
                    # print "classifier_validation.y"
                    # print classifier_validation.y
                    # classifier_validation.W = classifier.W
                    # print classifier_validation.W
                    # print "classifier_validation.W "
                    # classifier_validation.b = classifier.b
                    # print "classifier_validation.b"
                    # print classifier_validation.b
                    validation_losses.append(classifier_validation.errors())
                    # print "validation_losses"
                    # print validation_losses
                # print validation_losses
                this_validation_loss = np.mean(np.array(validation_losses))
                # print this_validation_loss
                # print "best_validation_loss"
                # print best_validation_loss


                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = []
                    for i in xrange(n_test_batches):
                        test_subset_x = test_set_x[i * batch_size: (i + 1) * batch_size]
                        test_subset_y = test_set_y[i * batch_size: (i + 1) * batch_size]
                        classifier_test.input = test_subset_x
                        classifier_test.y = test_subset_y
                        classifier_test.W = classifier.W
                        classifier_test.b = classifier.b
                        # print "classifier_test.input"
                        # print classifier_test.input
                        # print "classifier_test.y"
                        # print classifier_test.y
                        # print "classifier_test.W"
                        # print classifier_test.W
                        # print "classifier_test.b"
                        # print classifier_test.b
                        test_losses.append(classifier_test.errors())
                        # print "test_losses"
                        #  print test_losses
                    test_score = np.mean(np.array(test_losses))
                    # print "test_score"
                    # print test_score

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break
        epoch_num.append(epoch)
        valid_error.append(best_validation_loss)
        test_error.append(test_score)
        data_need = {"e":epoch_num,"v":valid_error, "t":test_error}
        cPickle.dump(data_need, open("ls_v_t_error.dat","wb"))
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # We can test it on some examples from test test
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    # test_set_x = test_set_x.get_value()

    classifier.input = test_set_x[:20]
    classifier.y = test_set_y[:20]
    classifier.compute_y_pred()
    # print classifier
    # print classifier.W
    # print classifier.computr

    print ("Predicted values for the first 10 examples in test set:")
    print classifier.y_pred
    print classifier.y

if __name__ == '__main__':
    sgd_optimization_mnist()

# predict()







