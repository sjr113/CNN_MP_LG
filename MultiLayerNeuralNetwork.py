# -*- coding:utf8 -*-
# coding=utf-8
__author__ = 'shen'

"""
A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.
.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5
"""

import os
import sys
import timeit

import numpy

from LG_MNIST import load_data, LogisticRegression

# start-snippet-1

# 隐含层
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation_type="tanh"):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        # Use the tricks to choose a suitable value for weight W and b
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=float
            )
            if activation_type == "sigmoid":
                W_values *= 4

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=float)

        self.W = W_values
        self.b = b_values

        # 计算隐含层的线性输出，但是当激活函数为tanh时，直接调用activation函数进行计算
        # 这里的output才是经过激活函数运算后该隐含层最终的输出
        # lin_output = numpy.dot(input, self.W) + self.b
        # self.output = (
        #     lin_output if activation_type is None
        #     else self.activation(activation_type, lin_output)
        # )
        # parameters of the model
        self.params = [self.W, self.b]

    # 为什么这里提示说最好把这个函数设置为静态函数呢？？？？？
    # def activation(self, function_type, z):
    #     if function_type ==  "tanh":
    #         return numpy.tanh(z)
    #     elif function_type == "sigmoid":
    #         return 1.0/(1 + numpy.exp(-z))
    #     else:
    #         raise ValueError('Your parameter is wrong! please choose sigmoid or tanh! ')

    # 该函数用来计算隐含层的z 和 a
    def forward_compute_z_a(self, input):
        self.z = numpy.dot(input, self.W) + self.b
        self.a = numpy.tanh(self.z)  # 隐含层的输出采用的是tanh函数

    # 该函数是计算该隐含层的delta值，由于该隐含层肯定不是最后一层，所以直接使用delta公式中求其他层的公式
    def back_delta(self, next_W, next_delta):
        tt = numpy.dot(next_delta, next_W.transpose())
        # 下式中的(1 - self.a ** 2)是指loss function（这里使用的是tanh，导数为1-（f(z）**2）对z的导数
        self.delta = tt * (1 - self.a ** 2)

    # 该函数用来更新w和b的值
    def back_update_w_b(self, a, learning_rate, L2_reg):
        delta_W = -1.0 * numpy.dot(a.transpose(), self.delta)/a.shape[0]
        delta_b = -1.0 * numpy.mean(self.delta, axis=0)
        self.W -= learning_rate * (L2_reg * self.W + delta_W)   # (784L, 500L)
        self.b -= learning_rate * delta_b   # (500L,)


# 输出层
class OutputLayer(LogisticRegression):
    def __init__(self, n_in, n_out):
        LogisticRegression.__init__(self, n_in, n_out)
        self.n_in = n_in
        self.n_out = n_out
    # 输出层计算最后的概率p_y_given_x的输出， x为最后一层的输入

    def forward_compute_p_y_given_x(self, x):
        self.exp_x_multiply_W_plus_b = numpy.exp(numpy.dot(x, self.W)+self.b)
        sigma = numpy.sum(self.exp_x_multiply_W_plus_b, axis=1)
        self.p_y_given_x = self.exp_x_multiply_W_plus_b/sigma.reshape(sigma.shape[0], 1)

    # 最后一层计算delta,计算公式为最后一层的delta公式
    def back_compute_delta(self, y):
        yy = numpy.zeros((y.shape[0], self.n_out))
        yy[numpy.arange(y.shape[0]), y] = 1.0
        self.delta = yy - self.p_y_given_x

    # 使用backpropagation算法更新W和b的值
    def back_update_w_b(self, a, learning_rate, L2_reg):
        delta_W = -1.0 * numpy.dot(a.transpose(), self.delta)/a.shape[0]
        delta_b = -1.0 * numpy.mean(self.delta, axis=0)
        self.W -= learning_rate * (L2_reg * self.W + delta_W)
        self.b -= learning_rate * delta_b
        # print "out_layer size of W and b   update"
        # print numpy.shape(self.W)   # (500L, 10L)
        # print numpy.shape(self.b)   # (10L,)

    def errors(self):
        # 选取概率值最大的类别作为最后的分类结果
        self.y_pred = numpy.argmax(self.p_y_given_x, axis=1)
        # print "self.y_pred"
        # print self.y_pred
        self.y_pred.resize(numpy.shape(self.y))
        # print self.y_pred
        # print self.y
        # print numpy.mean(numpy.not_equal(self.y_pred, self.y))
        return numpy.mean(numpy.not_equal(self.y_pred, self.y))


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng,  n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function

        self.hidden_layer_list = []
        self.hiddenLayer = HiddenLayer(
            rng=rng,  # 用来初始化权重参数W的随机数生成器
            # input=self.input,  # one minibatch
            n_in=n_in,  # 扇入
            n_out=n_hidden,  # 扇出，就是该隐藏层的神经元节点的个数
            activation_type="tanh"  # 给定激活函数的类型
        )

        # 当有多个隐含层的时候self.hidden_layer_list就可以保存多个隐藏层
        self.hidden_layer_list.append(self.hiddenLayer)
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer

        self.output_layer = OutputLayer(n_in=n_hidden, n_out=n_out)
        # self.logRegressionLayer.input = self.hidden_layer_list[0].a  # 隐含层的输出作为输出层的输入

        """
        # # end-snippet-2 start-snippet-3
        # # L1 norm ; one regularization option is to enforce L1 norm to
        # # be small
        # # 正则项L1
        # self.L1 = (
        #     abs(self.hiddenLayer.W).sum()
        #     + abs(self.logRegressionLayer.W).sum()
        # )
        # # 正则项L2
        # # square of L2 norm ; one regularization option is to enforce
        # # square of L2 norm to be small
        # self.L2_sqr = (
        #     (self.hiddenLayer.W ** 2).sum()
        #     + (self.logRegressionLayer.W ** 2).sum()
        # )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        # 由于最后一层是softmax层，故该层的negative_log_likelihood可以通过调用之前编写的Logistic Regression得到
        # self.negative_log_likelihood = (
        #     self.logRegressionLayer.negative_log_likelihood
        # )
        # same holds for the function computing the number of errors
        # self.errors = self.logRegressionLayer.errors
        # 计算输出层的errors由函数完成
        """
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        # 现在的multi layer NN 的模型由两层参数组成，分别是一层隐藏层和一层输出的softmax层
        self.params = self.hiddenLayer.params + self.output_layer.params
        # end-snippet-3

    def negative_log_likelihood(self):
        self.output_layer.input = self.hidden_layer_list[-1].a
        self.output_layer.y = self.y
        # print "self.hiddenLayer.W"
        # print self.hiddenLayer.W
        # print "self.output_layer.W"
        # print self.output_layer.W
        # 更新正则项的值，以便于计算cost function的值
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.output_layer.W ** 2).sum()
        # print "self.L2_sqr"
        # print self.L2_sqr
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.output_layer.W).sum()
        return self.output_layer.negative_log_likelihood()

    def errors(self):
        self.output_layer.input = self.hidden_layer_list[-1].a
        self.output_layer.y = self.y
        # print "self.output_layer.errors()"
        # print self.output_layer.errors()
        return self.output_layer.errors()

    # 使用前向传播算法分别计算隐含层和输出层的输出
    # 隐含层使用的激活函数是tanh
    # 输出层使用的激活函数是softmax
    def feedforward(self, input):
        xx = input
        for each in self.hidden_layer_list:
            each.forward_compute_z_a(xx)
            xx = each.a  # 计算隐含层的输出a   # 大小为20 * 500  (20L, 500L)
            # print "hidden_a"
            # print numpy.shape(xx)
            # print xx
        self.output_layer.forward_compute_p_y_given_x(xx)  # (20L, 10L)
        # print "output_layer  p_y_given_x"
        # print numpy.shape(self.output_layer.p_y_given_x)
        # print self.output_layer.p_y_given_x
        # 隐含层结束后计算输出层的p_y_given_x,输出层的输入即最后一个隐含层的输出

    # 实现backpropagation算法
    def backpropagation(self, x, y, learning_rate, L2_reg):
        self.output_layer.back_compute_delta(y)  # 计算最后一层的delta

        xx = self.hidden_layer_list[-1].a  # 最后一个隐含层的a  注意a=f(z)    (20L, 500L)
        # print "final hidden layer a"
        # print numpy.shape(xx)
        # print xx

        next_W = self.output_layer.W
        next_delta = self.output_layer.delta
        # 保存W和delta之后再对W进行更新

        len_hidden = len(self.hidden_layer_list)  # i为隐藏层的个数
        i = len_hidden
        j = len_hidden
        # print "len(self.hidden_layer_list)"
        # print len(self.hidden_layer_list)
        while i > 0:
            curr_hidden_lay = self.hidden_layer_list[i-1]  # 当前隐藏层，这是个Hidden_layer的对象
            curr_hidden_lay.back_delta(next_W, next_delta)   # 计算当前隐藏层的delta
            # if i > 1:
            #     xx = self.hidden_layer_list[i-2].a
            # else:
            #     xx = x  # 当i=1时，隐藏层的输入即给定的输入层的输入input
            # print "----------------"
            # curr_hidden_lay.back_update_w_b(xx, learning_rate,L2_reg)  # 更新隐藏层的W和b的值
            # print "@@@@@@@@@@@@@@@@"
            next_W = curr_hidden_lay.W
            next_delta = curr_hidden_lay.delta
            i -= 1
        self.output_layer.back_update_w_b(self.hidden_layer_list[-1].a, learning_rate, L2_reg)  # 更新最后一层outlayer的W和b
        while j > 0:
            curr_hidden_lay = self.hidden_layer_list[j-1]
            if j > 1:
                xx = self.hidden_layer_list[j-2].a
            else:
                xx = x  # 当i=1时，隐藏层的输入即给定的输入层的输入input
            curr_hidden_lay.back_update_w_b(xx, learning_rate,L2_reg)
            j -= 1


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]  # (50000L, 784L)  (50000L,)
    valid_set_x, valid_set_y = datasets[1]  # (10000L, 784L)   (10000L,)
    test_set_x, test_set_y = datasets[2]   #  (10000L, 784L)  (10000L,)

    # print "train"
    # print numpy.shape(train_set_x)
    # print numpy.shape(train_set_y)
    # print "valid"
    # print numpy.shape(valid_set_x)
    # print numpy.shape(valid_set_y)
    # print "test"
    # print numpy.shape(test_set_x)
    # print numpy.shape(test_set_y)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        n_in=28 * 28,
        n_hidden=n_hidden,  # 隐藏层节点的个数，这里应该是只有一个隐藏层
        n_out=10
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically

    # end-snippet-4
    classifier_validation = MLP(rng=rng, n_in=28 * 28, n_hidden=n_hidden, n_out=10)
    classifier_test = MLP(rng=rng, n_in=28 * 28, n_hidden=n_hidden, n_out=10)
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    # gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    # updates = [
    #     (param, param - learning_rate * gparam)
    #     for param, gparam in zip(classifier.params, gparams)
    # ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`

    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        jj = 0
        for minibatch_index in xrange(n_train_batches):
            jj += 1
            # minibatch_avg_cost = train_model(minibatch_index)
            # iteration number

            train_subset_x = train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] # (20L, 784L)
            train_subset_y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] # (20L,)

            classifier.input = train_subset_x  # 大小为20*784
            classifier.y = train_subset_y  # 大小为（20，）
            # print "before updating W"
            # print classifier.hidden_layer_list[-1].W
            # print classifier.output_layer.W
            # 更新权重W和b
            classifier.feedforward(classifier.input)  # 先进行前向传播运算，得到每一层的输出a
            # print classifier.hiddenLayer.a
            classifier.backpropagation(classifier.input, classifier.y, learning_rate, L2_reg=L2_reg)

            # 这里先只用L2的规范项，L1的暂时不用
            # minibatch_avg_cost = classifier.negative_log_likelihood()+ L1_reg * classifier.L1\
            #                      + L2_reg * classifier.L2_sqr
            # print "00000000000000000"
            # print numpy.shape(train_subset_x)
            # print numpy.shape(classifier.y)
            # print classifier.negative_log_likelihood()
            # print L2_reg
            # print classifier.L2_sqr
            minibatch_avg_cost = classifier.negative_log_likelihood() + L2_reg * classifier.L2_sqr
            # print "minibatch cost"
            # print minibatch_avg_cost
            if jj >2:
                pass
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                # validation_losses = [validate_model(i) for i
                #                      in xrange(n_valid_batches)]
                # this_validation_loss = numpy.mean(validation_losses)
                validation_losses = []
                # 考虑这里用W和b还是必须得用最终所有的参数params
                # 注意这里的W和b分开复制，因为每一层的W和b都是分别存放的
                # 输出层的W和b存在classifier.output_layer.W中，隐藏层的W分别存在classifier.hidden_layer_list[i-1]每个对象中
                classifier_validation.output_layer.W = classifier.output_layer.W
                classifier_validation.output_layer.b = classifier.output_layer.b
                kkk = len(classifier.hidden_layer_list)  # i为隐藏层的个数
                while kkk > 0:
                    curr_hidden_lay = classifier.hidden_layer_list[kkk-1]  # 当前隐藏层，这是个Hidden_layer的对象
                    (classifier_validation.hidden_layer_list[kkk-1]).W = curr_hidden_lay.W
                    (classifier_validation.hidden_layer_list[kkk-1]).b = curr_hidden_lay.b
                    kkk -= 1

                for i in xrange(n_valid_batches):
                    valid_subset_x = valid_set_x[i * batch_size: (i + 1) * batch_size]
                    valid_subset_y = valid_set_y[i * batch_size: (i + 1) * batch_size]
                    classifier_validation.input = numpy.array(valid_subset_x)
                    classifier_validation.y = numpy.array(valid_subset_y)

                    # 在计算errors之前应该是需要调用feedforward函数计算各层的输出，直到输出层，最后就可以得到errors
                    classifier_validation.feedforward(classifier_validation.input)
                    # 待改进：因为classifier_validation.feedforward函数已经计算过p_y_given_x，
                    # 但是LG_MNIST也计算了p_y_given_x！！！！！！！！！！
                    # classifier_validation.output_layer.input = classifier_validation.hidden_layer_list[-1].a
                    # print numpy.shape(classifier.output_layer.input)
                    # print numpy.shape(classifier.output_layer.W)
                    # print numpy.shape(classifier.output_layer.b)
                    # print "*********************************************************************"
                    validation_losses.append(classifier_validation.errors())

                this_validation_loss = numpy.mean(numpy.array(validation_losses))
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = []
                    for i in xrange(n_test_batches):
                        test_subset_x = test_set_x[i * batch_size: (i + 1) * batch_size]
                        test_subset_y = test_set_y[i * batch_size: (i + 1) * batch_size]
                        classifier_test.input = numpy.array(test_subset_x)
                        classifier_test.y = numpy.array(test_subset_y)
                        classifier_test.output_layer.W = classifier.output_layer.W
                        classifier_test.output_layer.b = classifier.output_layer.b
                        kkk = len(classifier.hidden_layer_list)  # i为隐藏层的个数
                        while kkk > 0:
                            curr_hidden_lay = classifier.hidden_layer_list[kkk-1]  # 当前隐藏层，这是个Hidden_layer的对象
                            (classifier_test.hidden_layer_list[kkk-1]).W = curr_hidden_lay.W
                            (classifier_test.hidden_layer_list[kkk-1]).b = curr_hidden_lay.b
                            kkk -= 1

                        classifier_test.feedforward(classifier_test.input)
                        # classifier_test.output_layer.input = classifier_test.hidden_layer_list[-1].a
                        test_losses.append(classifier_test.errors())

                    test_score = numpy.mean(numpy.array(test_losses))
                    # test it on the test set
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()

