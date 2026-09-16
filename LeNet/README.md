# Assignment1 LeNet实现CIFAR-10图像分类

## 说明

请先运行下面的命令以安装所需依赖，并按照wandb官方教程在环境变量中配置apikey

```shell
py -m pip install -r requirements.txt
```

接下来运行下面的命令训练和测试模型并查看结果

```shell
py lenet.py
```

在首次运行时会自动下载CIFAR-10数据集，之后每次运行时会自动检测CIFAR-10数据集是否存在或完整，防止重复下载占用资源
模型训练结果将保存在目录下的lenet.pt中，W&B日志也会保存在当前目录下

运行时程序会先检查cuda是否可用，若可用则会输出 device: cuda

## 本地运行结果

### [Momentum 0.99](https://wandb.ai/junelazarus-peking-university/lenet/runs/hsdd6vd9)

#### training accuracy

| epoch | epoch loss | validation accuracy |
| --- | --- | --- |
| 1 | 2.260 | 26.660% |
| 2 | 1.891 | 37.340% |
| 3 | 1.554 | 46.460% |
| 4 | 1.406 | 49.540% |
| 5 | 1.315 | 51.960% |
| 6 | 1.231 | 55.820% |
| 7 | 1.142 | 56.400% |
| 8 | 1.085 | 60.000% |
| 9 | 1.018 | 60.300% |
| 10 | 0.972 | 62.480% |

#### test accuracy

| class | accuracy |
| --- | --- |
| total | 62.52% |
| plane | 59.40 % |
| car | 73.80 % |
| bird | 48.30 % |
| cat | 44.60 % |
| deer | 60.30 % |
| dog | 50.50 % |
| frog | 63.40 % |
| horse | 70.10 % |
| ship | 77.40 % |
| truck | 77.40 % |

### [Momentum 0.9](https://wandb.ai/junelazarus-peking-university/lenet/runs/x41ibrqa)

#### training accuracy

| epoch | epoch loss | validation accuracy |
| --- | --- | --- |
| 1 | 2.303 | 15.160% |
| 2 | 2.296 | 16.060% |
| 3 | 2.263 | 21.600% |
| 4 | 2.219 | 27.560% |
| 5 | 2.014 | 31.160% |
| 6 | 1.906 | 32.920% |
| 7 | 1.812 | 35.720% |
| 8 | 1.730 | 38.580% |
| 9 | 1.659 | 40.100% |
| 10 | 1.600 | 41.660% |

#### test accuracy

| class | accuracy |
| --- | --- |
| total | 42.59% |
| plane | 45.00 % |
| car | 53.60 % |
| bird | 32.70 % |
| cat | 13.50 % |
| deer | 23.50 % |
| dog | 37.10 % |
| frog | 67.20 % |
| horse | 45.70 % |
| ship | 55.10 % |
| truck | 52.50 % |

### 结果分析

从当前条件下两组不同momentum的实验的W&B曲线可以看出，0.99组收敛明显比0.9组更快，且训练过程中振荡幅度也明显小于0.9组
结合sgd优化器的计算公式：
$$\begin{cases}v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t \\ \theta_{t+1} = \theta_t - \eta \cdot v_t \end{cases}$$
$v_t: 当前时刻的动量速度向量$
$\beta: 动量衰减系数$
$g_t: 当前批次的梯度$
$\eta: 学习率$
$\theta: 待更新参数的张量$
momentum决定了当前累计动量的保留比例和新随机梯度的采用比例，momentum越大则计算梯度偏向当前时间步前累积动量，越小则偏向当前批次采样梯度，可以看出更大的momentum能够削弱单批次随机梯度产生的噪声的影响，使得梯度下降更加平滑，进而更快收敛
