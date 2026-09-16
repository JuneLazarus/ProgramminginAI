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

可自行改动config中的超参数
