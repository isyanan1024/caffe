# 环境

ubuntu16.04

CUDA10

Cudnn7

# 安装

### 创建容器

该镜像已经装好了cuda10和cudnn7

```shell
nvidia-docker run -it --name yanan_caffe nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
```

###anaconda

版本为Anaconda3-2.4.1-Linux-x86_64.sh，该版本python为3.5，方便后面Boost的安装

```shell
Anaconda3-2.4.1-Linux-x86_64.sh
```

加入环境变量

```shell
export PATH=/root/anaconda3/bin/:$PATH
source ~/.bashrc
```

检查安装是否成功

```shell
python
```

### 换apt-get源

```shell
apt-get update
apt-get upgrade -y
```

### 安装库文件

以下命令每个可以运行两遍，确保每个库都完整的安装

```shell
apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler -y
```

```shell
apt-get install --no-install-recommends libboost-all-dev -y
```

```shell
apt-get install libopenblas-dev liblapack-dev libatlas-base-dev -y
```

```shell
apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev -y
```

```shell
apt-get install git cmake build-essential -y
```

###配置环境变量

```shell
vim ~/.bashrc
```

在最后添加下面内容

```shell
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

###安装Opencv

```shell
unzip OpenCV-3.4.5.zip
mv opencv-opencv-b7b8767/ opencv
cd opencv

mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j8
make install
```

检查是否安装成功

```shell
pkg-config --modversion opencv
```

出现版本号即成功

加入环境变量

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

###安装caffe

下载源码或者在服务器上拉压缩文件

```shell
git clone https://github.com/BVLC/caffe.git
```

```shell
unzip caffe-master.zip
mv caffe-master caffe
cd caffe

#将 Makefile.config.example 文件复制一份并更名为 Makefile.config 
cp Makefile.config.example Makefile.config
```

修改Makefile.config

- 修改：

```shell
#USE_CUDNN := 1
```

为：

```shell
USE_CUDNN := 1
```

- 修改：

```shell
#OPENCV_VERSION := 3
```

为：

```shell
OPENCV_VERSION := 3
```

- 修改(根据自己的GPU型号决定)：

```shell
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
                -gencode arch=compute_20,code=sm_21 \
                -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_52,code=sm_52 \
                -gencode arch=compute_60,code=sm_60 \
                -gencode arch=compute_61,code=sm_61 \
                -gencode arch=compute_61,code=compute_61
```

为：

```shell
CUDA_ARCH := -gencode arch=compute_61,code=compute_61
```

- 修改：

```shell
# PYTHON_LIBRARIES := boost_python3 python3.5m
```

为：

```shell
PYTHON_LIBRARIES := boost_python-py35 python3.5m
```

- 注释掉：

```shell
PYTHON_INCLUDE := /usr/include/python2.7 \
                /usr/lib/python2.7/dist-packages/numpy/core/include
```

为：

```shell
#PYTHON_INCLUDE := /usr/include/python2.7 \
                /usr/lib/python2.7/dist-packages/numpy/core/include
```

- 启用：

```shell
# ANACONDA_HOME := $(HOME)/anaconda
# PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
                # $(ANACONDA_HOME)/include/python2.7 \
                # $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include
```

为：

```shell
ANACONDA_HOME := /root/anaconda3
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
                $(ANACONDA_HOME)/include/python3.5m \
                $(ANACONDA_HOME)/lib/python3.5/site-packages/numpy/core/include
```

- 注释：

```shell
PYTHON_LIB := /usr/lib
```

为：

```shell
#PYTHON_LIB := /usr/lib
```

- 启用：

```shell
# PYTHON_LIB := $(ANACONDA_HOME)/lib
```

为：

```shell
PYTHON_LIB := $(ANACONDA_HOME)/lib
```

- 修改：

```shell
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
```

为：

```shell
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```

修改Makefile文件

- 修改：

```shell
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m
```

为：

```shell
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```

- 修改：

```shell
NVCCFLAGS +=-ccbin=$(CXX) -Xcompiler-fPIC $(COMMON_FLAGS)
```

为：

```shell
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
```

编译

```shell
make all -j32
 
#测试
make runtest -j8
```

可能会出现下面的错误：

```shell
liblapack.so.3: undefined symbol: gotoblas
```

解决方法：

```shell
update-alternatives --config libblas.so.3
update-alternatives --config liblapack.so.3
```

两个都选择atlas下的库即可

安装pycaffe

```shell
make pycaffe -j32
```

可能会出现这个错误：

```shell
fatal error: Python.h: No such file or directory
compilation terminated.
```

解决方法：检查Makefile.config中的Python路径有没有写对

添加环境变量：

```shell
echo export PYTHONPATH="/home/yanan/caffe/python" >> ~/.bashrc
source ~/.bashrc
```

安装ptotobuf

```shell
pip install protobuf==3.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证安装是否成功

```shell
python
import caffe
```

没报错即成功

全局使用caffe

```shell
export PATH=$PATH:/home/yanan/caffe/build/tools/
source ~/.bashrc
```

验证

```shell
caffe
```

出来相关信息即可

# MNIST

## caffe流程

准备数据-->定义Net-->配置Solver-->Run-->分析结果

### 准备数据

先下载准备好的数据，在caffe目录下运行

```shell
./data/mnist/get_mnist.sh
```

在~/caffe/data/mnist下有四个文件t10k-labels-idx1-ubyte、train-labels-idx1-ubyte、t10k-images-idx3-ubyte和train-images-idx3-ubyte

2. 将数据转换成LMDB格式

```shell
./examples/mnist/create_mnist.sh
```

在~/caffe/examples/mnist下多出mnist_test_lmdb和mnist_train_lmdb两个文件夹

### 定义Net

参考~/caffe/module_install_desc/mnist/mnist.prototxt

**注意**：需要修改两个source的数据集位置

### 配置Solver

参考~/caffe/module_install_desc/mnist/mnist_solver.prototxt

snapshot: 5000--每5000次保存一下模型

snapshot_prefix: "weights/snapshot"--模型保存的路径，snapshot为模型的开头名

## RUN

### 预估训练模型需要多久

```shell
caffe time -model ./caffe_mnist/hbk_mnist.prototxt -iterations 10 -gpu 0
```

-model：prototxt文件位置

-iterations：迭代次数，次数越多计算模型需要的时间越准确

-gpu：可选项，选择使用哪个GPU

### 训练

```shell
caffe train -solver mnist_solver.prototxt 2>&1 | tee train.log
```

## 分析结果

### 画网络结构图

- 安装graphviz：**apt-get install graphviz**
- 安装pydotplus：**pip install pydotplus**

```
python /home/yanan/caffe/python/draw_net.py mnist.prototxt --rankdir=LR net.png
Drawing net to net.png
```

**第一个参数**说明prototxt文件位置

**第二个参数**说明画图风格，可以选择TB(从上往下)、LR(从左往右)

**第三个参数**说明图片保存的名称

**注意**：prototxt文件中不要带中文，不然会报错

### 用网页画结构图

登陆http://ethereon.github.io/netscope/#/editor，复制prototxt内容到网页中，按shift+enter即可

## 分析训练数据

将分析数据的脚本模板复制一份

```shell
cp /home/yanan/caffe/tools/extra/plot_training_log.py.example /home/yanan/caffe/tools/extra/plot_training_log.py
```

根据序列号可以画出以下几种图

```shell
0: Test accuracy  vs. Iters
1: Test accuracy  vs. Seconds
2: Test loss  vs. Iters
3: Test loss  vs. Seconds
4: Train learning rate  vs. Iters
5: Train learning rate  vs. Seconds
6: Train loss  vs. Iters
7: Train loss  vs. Seconds
```

运行程序前需要保存训练的log，在训练的时候在指令后加上2>&1 | tee train.log

```shell
python /home/yanan/caffe/tools/extra/plot_training_log.py 6 pars.png train.log
```

**第一个参数**：需要画出哪种数据的表格，参照上面的序列号

**第二个参数**：表格的名称

**第三个参数**：训练的log日志

**注意事项**：

1. 出错：'dict_keys' object does not support indexing

修改plot_training_log.py第95行为：

```python
return list(markers.keys())[idx]
```

2. 出错：Invalid DISPLAY variable

在

```python
import matplotlib.markers as mks
```

下，添加：

```python
plt.switch_backend('agg')
```

3. 同时会把train.log分成train.log.train和train.log.test两个文本表格，可以更方便的观察数据

# Layer

数据层和网络层都是Layer的实现

# Net

是一堆Layer的组合

# Solver

是迭代算法

```python
solver = caffe.SGDSolver("")
```

通过读取Solver配置生成Solver对象，通过这个对象就可以获取到

```python
#训练用的net
solver.net
#测试用的net
solver.test_net[0]
```

solver.net还可以调用

```python
solver.net.forward()
```

可以通过

```python
solver.step(num)
```

表示算法需要迭代多少次，如果使用

```python
solver.solver()
```

则相当于运行了caffe train，会按照solver的配置文件进行迭代

```
solver.net.blobs['data']
```

blobs是caffe自定义的一种数据格式，可以理解为字典

```
solver.net.params['ip1'][0]
```

params表示权值，[0]表示权重，[1]表示偏置项

# pycaffe mnist

运行程序(在pycaffe_mnist文件夹下)

```python
python -i train.py
```

进入交互模式后，**输入**：

**solver.net**：<caffe._caffe.Net object at 0x7f0796e63ce8>

**solver.net.blobs**：输出字典格式的数据：

```shell
OrderedDict([('data', <caffe._caffe.Blob object at 0x7f0796e63be0>), ('label', <caffe._caffe.Blob object at 0x7f0796e63c38>), ('ip1', <caffe._caffe.Blob object at 0x7f0796e63ef8>), ('ip2', <caffe._caffe.Blob object at 0x7f0796e63d40>), ('loss', <caffe._caffe.Blob object at 0x7f0796e63e48>)])
```

**solver.net.blobs['ip1'].data.shape**：输出该节点的输入输出维度(64, 500)

**solver.net.params\['ip1'][0].data.shape**：输入对应节点的权重维度(500, 784)

**注意事项**：

1. 只在测试阶段打印准确率

```python
n.accu = L.Accuracy(n.ip2, n.label, include={'phase':caffe.TEST})
```

2. 使用GPU需添加如下代码

```python
caffe.set_mode_gpu()
caffe.set_device(0)
```

3. solvestate用来恢复训练，例如：

```shell
caffe train --solver=examples/test/solver.prototxt --snapshot=examples/test/test_100000.solverstate 
```

