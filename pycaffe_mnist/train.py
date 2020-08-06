# -*- coding: utf-8 -*-
# @Author: Yan An
# @Date: 2020-08-05 11:15:53
# @Last Modified by: Yan An
# @Last Modified time: 2020-08-05 11:54:50
# @Email: an.yan@intellicloud.ai
import caffe

from pylab import *
from caffe import layers as L
from caffe import params as P

caffe.set_mode_gpu()
caffe.set_device(0)

def net(dbfile, batch_size, mean_value=0):
    n = caffe.NetSpec()
    n.data, n.label=L.Data(source=dbfile, backend = P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(scale=0.00390625))
    n.ip1 = L.InnerProduct(n.data, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.accu = L.Accuracy(n.ip2, n.label, include={'phase':caffe.TEST})
    return n.to_proto()

with open( 'train.prototxt', 'w') as f:
    f.write(str(net( '/home/yanan/caffe/examples/mnist/mnist_train_lmdb', 64)))
with open('test.prototxt', 'w') as f:
    f.write(str(net('/home/yanan/caffe/examples/mnist/mnist_test_lmdb', 100)))

solver = caffe.SGDSolver('mnist_solver.prototxt')
solver.net.forward()
solver.test_nets[0].forward()

# solver.step(1)
solver.solve()