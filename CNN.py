# -*- coding:utf8 -*-
# coding=utf-8
__author__ = 'shen'
import numpy
import os
from scipy import signal
from scipy import fftpack
import sys
from LG_MNIST import load_data
from MultiLayerNeuralNetwork import HiddenLayer, OutputLayer
import timeit


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.pool_size = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit  隐藏层的输入， 大小为feature maps * filter height * filter width
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=numpy.float32)

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = numpy.zeros((filter_shape[0],), dtype=numpy.float32)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        # self.input = input

    # 卷积层的前向传导，计算卷积后的结果
    # 输入为卷积层的input，输出为经过卷积操作得到的z
    def forward_convolution(self, x):
        # print "x"
        # print numpy.shape(x)
        # print "W"
        # print numpy.shape(self.W)
        assert x.shape[1] == self.W.shape[1]
        self.z = numpy.zeros((x.shape[0], self.W.shape[0], x.shape[2]-self.W.shape[2]+1, x.shape[3]-self.W.shape[3]+1),
                             dtype=numpy.float32)
        i = 0
        while i < x.shape[0]:   # 对每一个样本
            j = 0
            while j < self.W.shape[0]:   # for each new filter
                tmp = numpy.zeros((x.shape[2]-self.W.shape[2]+1, x.shape[3]-self.W.shape[3]+1), dtype=numpy.float32)
                k = 0
                while k < self.W.shape[1]:   # for each input feature map
                    tmp += signal.correlate2d(x[i, k], self.W[j, k], 'valid')
                    k += 1
                self.z[i, j] = tmp + self.b[j]
                # add the bias term. Since the bias is a vector (1D array), we first
                # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
                # thus be broadcasted across mini-batches and feature map
                # width & height
                j += 1
            i += 1
        # 返回的z大小为：图像的个数*filter的个数
        # self.W的大小为：filter的个数*输入的feature map的个数
        # self.b的大小为filter的个数

    def forward_convolution_fft(self, x):
        assert x.shape[1] == self.W.shape[1]
        self.W = self.W[:, :, ::-1][:, :, :, ::-1]
        # rotate
        # i = 0
        # while i < self.W.shape[0]:
        #     j = 0
        #     while j < self.W.shape[1]:
        #         self.W[i, j] = numpy.rot90(self.W[i, j], 2)
        #         j += 1
        #     i += 1

        z_shape = (x.shape[0], self.W.shape[0], (x.shape[2]-self.W.shape[2]+1),
                   (x.shape[3]-self.W.shape[3]+1))
        self.z = numpy.zeros(z_shape, dtype=numpy.float32)

        # full_kernel_shape = (x.shape[2] + self.W.shape[2] - 1, x.shape[3] + self.W.shape[3] - 1)
        partial_kernel_shape = (x.shape[2], x.shape[3])

        fft_x = numpy.asarray(fftpack.fft2(x, partial_kernel_shape), numpy.complex64)
        fft_W = numpy.asarray(fftpack.fft2(self.W, partial_kernel_shape), numpy.complex64)

        i = 0
        while i < x.shape[0]:
            t = fft_x[i] * fft_W

            tt = numpy.sum(t, axis=1)
            ttt = numpy.asarray(numpy.real(fftpack.ifft2(tt)), numpy.float32)

            self.z[i] = ttt[:, (self.W.shape[2]-1) : x.shape[2], (self.W.shape[3]-1):x.shape[3]]
            i += 1

        self.a_before_pooling = numpy.tanh(self.z)

        # rotate back
        self.W = self.W[:, :, ::-1][:, :, :, ::-1]
        # i = 0
        # while i < self.W.shape[0]:
        #     j = 0
        #     while j < self.W.shape[1]:
        #         self.W[i, j] = numpy.rot90(self.W[i, j], 2)
        #         j += 1
        #     i += 1
        # return fft_x

    # pooling层的前向传导，计算pooling后的结果，这里使用的是max pooling
    # pooling层的前向传导，计算pooling后的结果，这里使用的是max pooling
    def feed_forward_pooling(self):
        n_rows = self.a_before_pooling.shape[2]/self.pool_size[0]  # 为pooling层的总行数
        n_cols = self.a_before_pooling.shape[3]/self.pool_size[1]  # 为pooling层的总列数
        self.beta = numpy.zeros_like(self.a_before_pooling)
        self.a = numpy.zeros((self.a_before_pooling.shape[0], self.a_before_pooling.shape[1], n_rows, n_cols))
        self.a_max_index_row = numpy.zeros((self.a_before_pooling.shape[0], self.a_before_pooling.shape[1],
                                            n_cols*n_rows), dtype=numpy.int)
        self.a_max_index_col = numpy.zeros((self.a_before_pooling.shape[0], self.a_before_pooling.shape[1],
                                            n_cols*n_rows), dtype=numpy.int)
        i = 0
        while i < self.a_before_pooling.shape[0]:  # for each sample
            j = 0
            while j < self.a_before_pooling.shape[1]:  # for each feature map
                k = 0
                while k < n_rows:
                    tt = self.a_before_pooling[i, j, (k * self.pool_size[0]):((k+1)*self.pool_size[0])]
                    l = 0
                    while l<n_cols:
                        tmp = tt[:, (l * self.pool_size[1]):((l+1)*self.pool_size[1])]
                        self.a[i, j, k, l] = numpy.max(tmp)  # 该patch内的最大值就是pooling的输出
                        inner_t_index = numpy.argmax(tmp) # 记录该patch内最大值的索引，该索引从0开始由行到列一次排列
                        inner_t_index_row = int(inner_t_index/self.pool_size[1])
                        inner_t_index_col = int(numpy.mod(inner_t_index, self.pool_size[1]))

                        self.a_max_index_row[i, j, k*n_cols + l] = k * self.pool_size[0] + inner_t_index_row
                        self.a_max_index_col[i, j, k*n_cols + l] = l * self.pool_size[1] + inner_t_index_col
                        self.beta[i, j, k * self.pool_size[0] + inner_t_index_row, l * self.pool_size[1] +
                                  inner_t_index_col] = 1
                        l += 1
                    k += 1
                j += 1
            i += 1

    def split_max_pooling(self, _in_tensor):

        t_shape = _in_tensor.shape
        n_rows = int(numpy.floor(t_shape[2]/self.pool_size[0]))
        n_cols = int(numpy.floor(t_shape[3]/self.pool_size[1]))

        tmp = numpy.asarray(numpy.split(_in_tensor, n_cols, axis=3))
        return numpy.asarray(numpy.split(tmp, n_rows, axis=3))

    def concatenate_max_pooling(self, _in_tensor):

        return numpy.concatenate(numpy.concatenate(_in_tensor, axis=3), axis=3)

    def feed_forward_pooling_fft(self):
        # temporarily ignore the boundary handling
        assert numpy.mod(self.a_before_pooling.shape[2], self.pool_size[0]) == 0
        assert numpy.mod(self.a_before_pooling.shape[3], self.pool_size[1]) == 0

        n_rows = self.a_before_pooling.shape[2]/self.pool_size[0]  # 为pooling层的总行数
        n_cols = self.a_before_pooling.shape[3]/self.pool_size[1]  # 为pooling层的总列数
        self.a = numpy.zeros((self.a_before_pooling.shape[0], self.a_before_pooling.shape[1], n_rows, n_cols))

        tmp = self.split_max_pooling(self.a_before_pooling)
        tmp1 = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape[3], self.pool_size[0] * self.pool_size[1]))

        tmp2 = tmp1.max(axis=4)
        self.id_max = tmp1.argmax(axis=4)
        # tmp1[:, :, :, :, self.id_max] = 0
        tmp3 = tmp2.reshape((tmp2.shape[0], tmp2.shape[1], tmp2.shape[2], tmp2.shape[3], 1, 1))
        self.a = self.concatenate_max_pooling(tmp3)

    # 根据pooling层的delta计算卷积层的delta
    def compute_conv_delta_from_pooling_layer(self, next_delta):
        # 首先生成unsample
        # 1.把存储最大值的索引生成一个等大的矩阵
        self.delta_conv = numpy.zeros(self.a_before_pooling.shape, dtype=numpy.float32)
        i = 0
        while i<self.a_before_pooling.shape[0]:  # for each sample
            j = 0
            while j<self.a_before_pooling.shape[1]:  # for each feature map conv layer
                # k = 0
                # while k < next_delta.shape[1]:  # for each feature map in the next layer
                #     unsample = scipy.linalg.kron(numpy.array(self.delta_pooling[i, k]), numpy.array([[1, 1], [1, 1]]))
                #     self.delta_conv[i, j] = unsample * self.beta[i, j]
                #     k += 1
                self.delta_conv[i, j, self.a_max_index_row[i, j], self.a_max_index_col[i, j]] \
                        = self.delta_pooling[i, j].reshape(numpy.prod(self.delta_pooling[i, j].shape),)
                j += 1
            i += 1

    def compute_conv_delta_from_pooling_layer_fft(self, next_delta):
        t_shape = self.a_before_pooling.shape

        tt_shape = (int(t_shape[2]/self.pool_size[0]), int(t_shape[3]/self.pool_size[1]), t_shape[0],
                        t_shape[1], self.pool_size[0] * self.pool_size[1])

        tmp = numpy.zeros((numpy.prod(tt_shape[:4]), tt_shape[4]), dtype=numpy.float32)

        t = numpy.asarray(numpy.split(self.delta_pooling, self.delta_pooling.shape[3], axis=3))
        tt = numpy.asarray(numpy.split(t, self.delta_pooling.shape[2], axis=3))
        ttt = tt.reshape(numpy.prod(self.delta_pooling.shape),)

        tmp[numpy.arange(numpy.prod(tt_shape[:4])), self.id_max.reshape(numpy.prod(self.id_max.shape))] = ttt

        tmp2 = tmp.reshape((tt_shape[0], tt_shape[1], tt_shape[2], tt_shape[3], self.pool_size[0], self.pool_size[1]))

        self.delta_conv = self.concatenate_max_pooling(tmp2)

    # 根据卷积层的delta 计算pooling层的delta
    def compute_pool_layer_delta_from_conv_layer(self, next_delta, next_W):
        self.delta_pooling = numpy.zeros(self.a.shape, dtype=numpy.float32)
        i = 0
        while i<self.a.shape[0]:  # for each sample
            j = 0
            while j<self.a.shape[1]:  # for each feature map pooling layer
                dummy = numpy.zeros((self.a.shape[2], self.a.shape[3]), dtype=numpy.float32)
                k = 0
                while k<next_delta.shape[1]:  # for each feature map in the next layer
                    dummy += signal.convolve2d(next_delta[i, k], next_W[k,j], "full")
                    k += 1
                self.delta_pooling[i, j] = dummy
                j += 1
            i += 1

    def compute_pool_layer_delta_from_conv_layer_fft(self, next_delta, next_W):
        self.delta_pooling = numpy.zeros(self.a.shape, dtype=numpy.float32)

        full_kernel_shape = (next_delta.shape[2] + next_W.shape[2] - 1,
                                 next_delta.shape[3] + next_W.shape[3] - 1)

        fft_next_delta = numpy.asarray(fftpack.fft2(next_delta, full_kernel_shape), dtype=numpy.complex64)
        fft_next_W = numpy.asarray(fftpack.fft2(next_W, full_kernel_shape), dtype=numpy.complex64)

        i = 0
        for i in numpy.arange(self.a.shape[0]): # for each sample
            t = fft_next_delta[i].reshape(next_delta.shape[1], 1, full_kernel_shape[0], full_kernel_shape[1])
            tt = t * fft_next_W
            ttt = numpy.sum(tt, axis=0)
            self.delta_pooling[i] = numpy.asarray(numpy.real(fftpack.ifft2(ttt)), dtype=numpy.float32)

    def compute_pooling_layer_delta_from_hidden_layer(self, next_W, next_delta):
        tt = numpy.dot(next_delta, next_W.transpose())

        ttt = tt.reshape(self.a.shape)
        # print numpy.shape(ttt)
        # 下式中的(1 - self.a ** 2)是指loss function（这里使用的是tanh，导数为1-（f(z）**2）对z的导数
        tttt = ttt * (1 - self.a ** 2)  # (20L, 50L, 4L, 4L)
        self.delta_pooling = numpy.zeros((self.a.shape[0], self.a.shape[1]))
        # i = 0
        # while i<self.a.shape[0]:
        #     j = 0
        #     while j<self.a.shape[1]:
        #         self.delta_pooling[i, j] = sum(sum(tttt[i, j, :, :]))
        #         j += 1
        #     i += 1
        self.delta_pooling = tttt
        # print numpy.shape(self.delta_pooling)  # (20L, 50L, 4L, 4L)

    def update_W_b(self, n_samples, x, learning_rate, L2_reg):
        i = 0
        while i < self.delta_conv.shape[0]:
            j = 0
            while j < self.delta_conv.shape[1]:
                self.delta_conv[i, j] = numpy.rot90(self.delta_conv[i, j], 2)
                j += 1
            i += 1

        n_samples = x.shape[0]
        delta_W = numpy.zeros(self.W.shape, dtype=numpy.float32)

        j = 0
        while j<self.W.shape[0]: # for each new filter group
            k = 0
            while k<self.W.shape[1]: # for each old feature map
                i = 0
                while i< n_samples:  # for each sample
                    # print "update_W_b"
                    # print numpy.shape(x)  # (20L, 1L, 28L, 28L)
                    # print numpy.shape(self.delta_conv)  # (20L, 20L, 24L, 24L)
                    delta_W[j, k] += signal.convolve2d(x[i, k], self.delta_conv[i, j], "valid")
                    i += 1
                k += 1
            j += 1

        delta_b = numpy.sum(numpy.sum(numpy.sum(self.delta_conv, axis=0), axis=1), axis=1)

        # print "delta_b"
        # print numpy.shape(delta_b)
        # print "delta_W"
        # print numpy.shape(delta_W)
        delta_W = -1.0 * delta_W / n_samples
        self.W -= learning_rate * (L2_reg * self.W + delta_W)

        delta_b = numpy.zeros(self.b.shape, dtype=numpy.float32)

        t = numpy.sum(self.delta_conv, axis=0)
        t = numpy.sum(t, axis=1)
        delta_b = numpy.sum(t, axis=1)

        delta_b = -1.0 * delta_b / n_samples
        self.b -= learning_rate * delta_b

        # rotate back
        i = 0
        while i < self.delta_conv.shape[0]:
            j = 0
            while j < self.delta_conv.shape[1]:
                self.delta_conv[i, j] = numpy.rot90(self.delta_conv[i, j], 2)
                j += 1
            i += 1

    def update_W_b_fft(self, n_samples, x, learning_rate, L2_reg):
        # for i in numpy.arange(self.delta_conv.shape[0]):
        #     for j in numpy.arange(self.delta_conv.shape[1]):
        #         self.delta_conv[i, j] = numpy.rot90(self.delta_conv[i, j], 2)
        # rotate
        self.delta_conv = self.delta_conv[:,:,::-1][:,:,:,::-1]

        n_samples = x.shape[0]

        partial_kernel_shape = (x.shape[2], x.shape[3])

        fft_x = numpy.asarray(fftpack.fft2(x, partial_kernel_shape), dtype=numpy.complex64)

        # b = timeit.default_timer()

        fft_delta_conv = numpy.asarray(fftpack.fft2(self.delta_conv, partial_kernel_shape), dtype=numpy.complex64)
        # e = timeit.default_timer()
        # print('fft %f s' % (e-b))
        fft_x_reshape = fft_x.reshape(n_samples, 1, x.shape[1], partial_kernel_shape[0], partial_kernel_shape[1])
        fft_delta_conv_reshape = fft_delta_conv.reshape(n_samples, self.delta_conv.shape[1], 1, partial_kernel_shape[0], partial_kernel_shape[1])


        t = fft_x_reshape * fft_delta_conv_reshape
        tt = numpy.sum(t, axis=0)
        ttt = numpy.asarray(numpy.real(fftpack.ifft2(tt)), dtype=numpy.float32)


        delta_W = ttt[:, :, (self.delta_conv.shape[2]-1) : x.shape[2], (self.delta_conv.shape[3]-1):x.shape[3]]


        # exit()

        # self.delta_W_ = delta_W

        # j = 0
        # while j < self.W.shape[0]: # for each new filter group
        #     k = 0
        #     while k < self.W.shape[1]: # for each old feature map
        #         i = 0
        #         while i < n_samples: # for each sample
        #             delta_W[j, k] += signal.convolve2d(x[i, k], self.delta_conv[i, j], 'valid')
        #             i += 1
        #         k += 1
        #     j += 1

        delta_W = -1.0 * delta_W / n_samples
        # delta_W = -1.0 * delta_W
        self.W -= learning_rate * (L2_reg * self.W + delta_W)

        # print('norm and mean of the delta W in conv-pool layer: %f, %f' % (numpy.sqrt(numpy.sum(delta_W**2)),
        #      numpy.mean(numpy.abs(delta_W))))

        i = 0
        delta_b = numpy.zeros(self.b.shape, dtype=numpy.float32)

        t = numpy.sum(self.delta_conv, axis=0)
        t = numpy.sum(t, axis=1)
        delta_b = numpy.sum(t, axis=1)

        delta_b = -1.0 * delta_b / n_samples
        self.b -= learning_rate * delta_b

        # rotate back
        self.delta_conv = self.delta_conv[:,:,::-1][:,:,:,::-1]
        # i = 0
        # while i < self.delta_conv.shape[0]:
        #     j = 0
        #     while j < self.delta_conv.shape[1]:
        #         self.delta_conv[i, j] = numpy.rot90(self.delta_conv[i, j], 2)
        #         j += 1
        #     i += 1


class cnn(object):
    def __init__(self, rng, nkerns, batch_size):

        # 第0层是卷积层加pooling层，20个feature map.5*5的卷积，2*2的pooling size
        self.layer0 = LeNetConvPoolLayer(
            rng,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        # 第一层也是卷积层+pooling层，feature map的个数为50， 5*5的卷积，2*2的pooling size
        self.layer1 = LeNetConvPoolLayer(
            rng,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        # layer2_input = layer1.output.flatten(2)
        # 将layer1的输出展平：(500, 50 * 4 * 4) = (500, 800)
        # construct a fully-connected sigmoidal layer
        # 第二层是hidden layer 输入大小为50 * 4 * 4
        self.layer2 = HiddenLayer(
            rng,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activation_type="tanh"
        )
        # layer2.input = layer2_input
        # classify the values of the fully-connected sigmoidal layer
        # 第三层是softmax层， 输入为500个样本（一个batch size）， 输出为10（MNIST数据集一共10类）
        self.layer3 = OutputLayer(n_in=500, n_out=10)

    def negative_log_likelihood(self, y):
        self.layer3.input = self.layer2.a
        self.layer3.y = y
        # print "self.hiddenLayer.W"
        # print self.hiddenLayer.W
        # print "self.output_layer.W"
        # print self.output_layer.W
        # 更新正则项的值，以便于计算cost function的值
        self.layer3.L2_sqr = (self.layer2.W ** 2).sum() + (self.layer3.W ** 2).sum()
        # print "self.L2_sqr"
        # print self.L2_sqr
        self.layer3.L1 = abs(self.layer2.W).sum() + abs(self.layer3.W).sum()
        return self.layer3.negative_log_likelihood()

    def errors(self, y):
        self.layer3.input = self.layer2.a
        self.layer3.y = y
        # print "self.output_layer.errors()"
        # print self.output_layer.errors()
        return self.layer3.errors()

    # 使用前向传播算法分别计算隐含层和输出层的输出
    # 隐含层使用的激活函数是tanh
    # 输出层使用的激活函数是softmax
    def feedforward(self, input):
        xx = input
        # 1. 采用前向传导算法首先计算第0层的输出
        self.layer0.forward_convolution(input)
        # self.layer0.forward_convolution_fft(input)
        # 激活函数使用tanh函数，但是可以先做pooling运算，再做tanh...可改进！！！！
        self.layer0.a_before_pooling = numpy.tanh(self.layer0.z)
        self.layer0.feed_forward_pooling()
        # self.layer0.feed_forward_pooling_fft()
        # 此时的输出为self.a 为pooling层的输出
        # print "feed_forward"
        # print numpy.shape(self.layer0.a)
        # 2. 采用前向传导算法计算第1层的输出
        self.layer1.forward_convolution(self.layer0.a)   # (20L, 20L, 12L, 12L)
        # self.layer1.forward_convolution_fft(self.layer0.a)
        self.layer1.a_before_pooling = numpy.tanh(self.layer1.z)  # 同上，可改进！！！！
        self.layer1.feed_forward_pooling()
        # self.layer1.feed_forward_pooling_fft()
        # print numpy.shape(self.layer1.a)  # (20L, 50L, 4L, 4L)
        # 3.计算隐藏层的输出
        # 注意首先要把卷积层+pooling层的输出展平才能作为隐藏层的输入
        i = 0
        layer2_input = numpy.zeros((self.layer1.a.shape[0], self.layer1.a.shape[1]*self.layer1.a.shape[2]*self.layer1.a.shape[3]))
        while i <self.layer1.a.shape[0]:
            # print numpy.shape(self.layer1.a[i, :])
            layer2_input[i, :] = self.layer1.a[i, :].flatten()  # 展开时应该怎样展开！！！
            i += 1

        # print numpy.shape(layer2_input)
        self.layer2.forward_compute_z_a(layer2_input)  # (20L, 800L)

        # 4. 计算最后一层softmax层的输出
        # 隐含层结束后计算输出层的p_y_given_x,输出层的输入即最后一个隐含层的输出
        self.layer3.forward_compute_p_y_given_x(self.layer2.a)  # (20L, 10L)

    def back_propogation(self, x, y, learning_rate, L2_reg):
        # 1. 计算最后一层(输出层)的delta
        # start_time = timeit.default_timer()
        self.layer3.back_compute_delta(y)
        # end_time = timeit.default_timer()
        # print "time of output layer delta:" + str(end_time-start_time)
        # xx = self.layer2.a  # 隐含层的a  注意a=f(z)
        # 保存输出层的W和delta之后再对W进行更新
        next_W = self.layer3.W   # (500L, 10L)
        next_delta = self.layer3.delta  # (20L, 10L)
        # print "layer3"
        # print numpy.shape(next_W)
        # print numpy.shape(next_delta)
        # 2. 计算当前隐藏层的delta
        # start_time = timeit.default_timer()
        self.layer2.back_delta(next_W, next_delta)
        # end_time = timeit.default_timer()
        # print "time of hidden layer delta:" + str(end_time-start_time)
        next_W = self.layer2.W  # (800L, 500L)
        next_delta = self.layer2.delta  # (20L, 500L)
        # print "layer2"
        # print numpy.shape(next_W)
        # print numpy.shape(next_delta)
        # 3. 计算倒数第一个卷积层+pooling层的delta
        # 注意：从hidden层到pooling层没有W，但是需要计算delta
        # 计算从hidden layer层到pooling层的delta
        # start_time = timeit.default_timer()
        self.layer1.compute_pooling_layer_delta_from_hidden_layer(next_W, next_delta)
        # end_time = timeit.default_timer()
        # print "time of compute_pooling_layer_delta_from_hidden_layer:" + str(end_time-start_time)
        next_delta = self.layer1.delta_pooling
        # print "layer1"
        # # print numpy.shape(next_W)
        # print numpy.shape(next_delta)  # (20L, 50L, 4L, 4L)
        # start_time = timeit.default_timer()
        self.layer1.compute_conv_delta_from_pooling_layer(next_delta)
        # self.layer1.compute_conv_delta_from_pooling_layer_fft(next_delta)
        # end_time = timeit.default_timer()
        # print "time of compute_conv_delta_from_pooling_layer:" + str(end_time-start_time)
        next_W = self.layer1.W
        next_delta = self.layer1.delta_conv
        # print "layer1"
        # print numpy.shape(next_W)  # (50L, 20L, 5L, 5L)
        # print numpy.shape(next_delta)  # (20L, 50L, 8L, 8L)
        # 4. 计算倒数第二个卷积层+pooling层的delta
        # start_time = timeit.default_timer()
        self.layer0.compute_pool_layer_delta_from_conv_layer(next_delta, next_W)
        # self.layer0.compute_pool_layer_delta_from_conv_layer_fft(next_delta, next_W)
        # end_time = timeit.default_timer()
        # print "time of compute_pool_layer_delta_from_conv_layer:" + str(end_time-start_time)
        # next_W = self.layer0.W
        next_delta = self.layer0.delta_pooling
        # print "layer0"
        # print numpy.shape(next_W)
        # print numpy.shape(next_delta)  # (20L, 20L, 12L, 12L)
        # start_time = timeit.default_timer()
        self.layer0.compute_conv_delta_from_pooling_layer(next_delta)
        # self.layer0.compute_conv_delta_from_pooling_layer_fft(next_delta)
        # end_time = timeit.default_timer()
        # print "time of compute_conv_delta_from_pooling_layer:" + str(end_time-start_time)
        next_W = self.layer0.W
        next_delta = self.layer0.delta_conv
        # print "layer0"
        # print numpy.shape(next_W)  # (20L, 1L, 5L, 5L)
        # print numpy.shape(next_delta)  # (20L, 20L, 24L, 24L)
        # 下面开始更新W
        self.layer0.update_W_b(n_samples=x.shape[0], x=x, learning_rate=learning_rate,L2_reg=L2_reg)
        # self.layer0.update_W_b_fft(n_samples=x.shape[0], x=x, learning_rate=learning_rate,L2_reg=L2_reg)
        self.layer1.update_W_b(n_samples=x.shape[0], x=self.layer0.a, learning_rate=learning_rate,L2_reg=L2_reg)
        # self.layer1.update_W_b_fft(n_samples=x.shape[0], x=self.layer0.a, learning_rate=learning_rate,L2_reg=L2_reg)
        layer2_input = numpy.zeros((self.layer1.a.shape[0], self.layer1.a.shape[1]*self.layer1.a.shape[2]*self.layer1.a.shape[3]))
        i = 0
        while i < self.layer1.a.shape[0]:
            # print numpy.shape(self.layer1.a[i, :])
            layer2_input[i, :] = self.layer1.a[i, :].flatten()  # 展开时应该怎样展开！！！
            i += 1
        self.layer2.back_update_w_b(a=layer2_input, learning_rate=learning_rate, L2_reg=L2_reg)
        self.layer3.back_update_w_b(self.layer2.a, learning_rate=learning_rate, L2_reg=L2_reg)  # 更新最后一层outlayer的W和b


def test_cnn(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=500, n_hidden=500):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]  # (50000L, 784L)  (50000L,)
    valid_set_x, valid_set_y = datasets[1]  # (10000L, 784L)   (10000L,)
    test_set_x, test_set_y = datasets[2]   #  (10000L, 784L)  (10000L,)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    rng = numpy.random.RandomState(1234)

    nkerns = [20, 50]

    # the cost we minimize during training is the NLL of the model
    # cost = layer3.negative_log_likelihood(y)

    # 初始化CNN分类器
    classifier = cnn(rng, nkerns, batch_size)

    classifier_validation = cnn(rng, nkerns, batch_size)
    classifier_test = cnn(rng, nkerns, batch_size)
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
        start_time_epoch = timeit.default_timer()
        print "-----------------------------------------------------"
        print "epoch :" + str(epoch)
        epoch = epoch + 1
        # jj = 0
        for minibatch_index in xrange(n_train_batches):
            # jj += 1
            # minibatch_avg_cost = train_model(minibatch_index)
            # iteration number

            train_subset_x = train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] # (20L, 784L)
            train_subset_y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] # (20L,)

            classifier_input = train_subset_x.reshape(batch_size, 1, 28, 28)  # 大小为20*784
            classifier_y = train_subset_y  # 大小为（20，）

            # 更新权重W和b
            classifier.feedforward(classifier_input)  # 先进行前向传播运算，得到每一层的输出a
            # print classifier.hiddenLayer.a
            classifier.back_propogation(classifier_input, classifier_y, learning_rate, L2_reg=L2_reg)

            # 这里先只用L2的规范项，L1的暂时不用
            # minibatch_avg_cost = classifier.negative_log_likelihood()+ L1_reg * classifier.L1\
            #                      + L2_reg * classifier.L2_sqr

            minibatch_avg_cost = classifier.negative_log_likelihood(train_subset_y) + L2_reg * classifier.layer3.L2_sqr
            print "minibatch_avg_cost" + str(minibatch_avg_cost)

            iter = (epoch - 1) * n_train_batches + minibatch_index
            # if jj >2:
            #     pass
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                # validation_losses = [validate_model(i) for i
                #                      in xrange(n_valid_batches)]
                # this_validation_loss = numpy.mean(validation_losses)
                validation_losses = []
                # 考虑这里用W和b还是必须得用最终所有的参数params
                # 注意这里的W和b分开复制，因为每一层的W和b都是分别存放的
                # 输出层的W和b存在classifier.output_layer.W中，隐藏层的W分别存在classifier.hidden_layer_list[i-1]每个对象中
                classifier_validation.layer3.W = classifier.layer3.W
                classifier_validation.layer3.b = classifier.layer3.b
                classifier_validation.layer2.W = classifier.layer2.W
                classifier_validation.layer2.b = classifier.layer2.b
                classifier_validation.layer1.W = classifier.layer1.W
                classifier_validation.layer1.b = classifier.layer1.b
                classifier_validation.layer0.W = classifier.layer0.W
                classifier_validation.layer0.b = classifier.layer0.b

                for i in xrange(n_valid_batches):
                    valid_subset_x = valid_set_x[i * batch_size: (i + 1) * batch_size]
                    valid_subset_y = valid_set_y[i * batch_size: (i + 1) * batch_size]

                    classifier_validation_input = valid_subset_x.reshape(batch_size, 1, 28, 28)  # 大小为20*784
                    classifier_validation_y = valid_subset_y  # 大小为（20，）
                    # 在计算errors之前应该是需要调用feedforward函数计算各层的输出，直到输出层，最后就可以得到errors
                    classifier_validation.feedforward(classifier_validation_input)
                    # 待改进：因为classifier_validation.feedforward函数已经计算过p_y_given_x，
                    # 但是LG_MNIST也计算了p_y_given_x！！！！！！！！！！
                    # classifier_validation.output_layer.input = classifier_validation.hidden_layer_list[-1].a

                    validation_losses.append(classifier_validation.errors(classifier_validation_y))

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
                        classifier_test_input = test_subset_x.reshape(batch_size, 1, 28, 28)  # 大小为20*784
                        classifier_test_y = test_subset_y  # 大小为（20，）
                        classifier_test.layer3.W = classifier.layer3.W
                        classifier_test.layer3.b = classifier.layer3.b
                        classifier_test.layer2.W = classifier.layer2.W
                        classifier_test.layer2.b = classifier.layer2.b
                        classifier_test.layer1.W = classifier.layer1.W
                        classifier_test.layer1.b = classifier.layer1.b
                        classifier_test.layer0.W = classifier.layer0.W
                        classifier_test.layer0.b = classifier.layer0.b

                        classifier_test.feedforward(classifier_test_input)
                        test_losses.append(classifier_test.errors(classifier_test_y))

                    test_score = numpy.mean(numpy.array(test_losses))

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
        end_time_epoch = timeit.default_timer()
        print "Time: " + str(end_time_epoch-start_time_epoch)
        print "#############################################################"
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

test_cnn()












