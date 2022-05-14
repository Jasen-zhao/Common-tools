import time
import numpy as np
import torch
from IPython import display
import torchvision
import torch.nn as nn
from .pyplot import *
from .creat_dataset import MyDataset

from tqdm import tqdm
import wandb
import random



#计时模块
class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()
    def start(self):
        """Start the timer."""
        self.tik = time.time()
    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)
    def sum(self):
        """Return the sum of time."""
        return sum(self.times)
    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()





#图像绘制模块
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()





class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        display.set_matplotlib_formats('svg')
        self.fig, self.axes = subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)





#累加器
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]





#准确率计算
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)#匿名函数
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
def accuracy(y_hat, y):
    """Compute the number of correct predictions.
    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))





#训练一个batch
def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13).
    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum





#推理评估准确率
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.
    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]





#查找gpu
def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]



#设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)  #设置CPU生成随机数的种子
     torch.cuda.manual_seed_all(seed)  #给所有GPU设置随机数种子
     np.random.seed(seed)  #numpy设置随机数种子
     random.seed(seed)  #python设置随机数种子
     torch.backends.cudnn.deterministic = True  #确定每次返回的默认算法



#模型训练
def train(net, train_iter, test_iter, loss, trainer, num_epochs,devices=try_all_gpus(),
        tqdm_open=False,plot_training=False,wandb_open=False):
    """
    trainer:优化器,SGD
    devices=try_all_gpus()即如果不指定,将自己查找可用的GPU
    tqdm_open:是否打开进度条
    plot_training:是否绘制训练过程图,一般进度条和训练过程图只用一个。
    wandb_open:是否打开wandb
    """
    timer, num_batches = Timer(), len(train_iter)
    
    if plot_training:
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                                legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    if wandb_open:
        wandb.init(project="mobilenet v3",name="mobilev3_p1")
        config = wandb.config
        config.batch_size = 32
        config.test_batch_size = 32
        config.epochs = num_epochs
        config.lr = 0.0001
        config.momentum = 0.1
        config.no_cuda = False
        config.seed = 42

        wandb.watch(net, log="all",log_freq=5)

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = Accumulator(4)
        if tqdm_open:
            #初始化一个进度条
            progress_bar =tqdm(total=num_batches, leave=True, desc='train epoch '+str(epoch),
                dynamic_ncols=True)
        
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            #这里得到的l是一个batch里累计的损失，acc是一个batch里正确的数量
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())

            if tqdm_open:
                progress_bar.set_postfix({'Acc' : '{0:1.2f}'.format(metric[1] / metric[3]),
                    'loss' : '{0:1.2f}'.format(metric[0] / metric[2])})#更新进度条
                progress_bar.update()
            timer.stop()

            if plot_training:
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                (metric[0] / metric[2], metric[1] / metric[3],
                                None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        if plot_training:
            animator.add(epoch + 1, (None, None, test_acc))
        
        if tqdm_open:
            progress_bar.close()#关闭进度条

        if wandb_open:
            wandb.log({'epoch': epoch, 'train loss': metric[0] / metric[2],
                'train acc':metric[1] / metric[3],'test acc':test_acc})
    
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    
    if wandb_open:
        torch.save(net.state_dict(), "mobilev3_p1.pth")
        wandb.save('mobilev3_p1.pth')





#训练过程举例
if __name__ == '__main__':
    #数据增强
    normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),#转到256
        torchvision.transforms.RandomRotation(90),#旋转（-90，90）
        torchvision.transforms.CenterCrop(224),#从中央裁剪224
        torchvision.transforms.ToTensor(),#转成tensor数据
        normalize])


    #读取创建的数据集
    train_dataset = MyDataset(csv_file='../datasets/strawberry/preliminary_first/train.csv',
                            root_dir='../datasets/strawberry/preliminary_first/train/',
                            transform=train_augs)#其实也可以transform为数据预处理和数据增强


    #初始化模型
    mobilenet_v3_small= torchvision.models.mobilenet_v3_small(pretrained=True)
    mobilenet_v3_small.classifier[3]=nn.Linear(mobilenet_v3_small.classifier[3].in_features, 4)
    setup_seed(20)# 设置随机数种子
    nn.init.xavier_uniform_(mobilenet_v3_small.classifier[3].weight)#初始化
    

    #模型微调
    def train_fine_tuning(net, learning_rate, batch_size=32, num_epochs=10,param_group=True):
        #param_group=True：意味着学习率划分，输出层中的模型参数将使用十倍的学习率
        
        train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
        test_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
        
        devices = [torch.device('cuda:5'),torch.device('cuda:6')]#多gpu训练，[torch.device('cuda:5')]代表单gpu训练
        loss = nn.CrossEntropyLoss(reduction="none")
        
        if param_group:
            #排除更改的参数，可以用print(list(net.named_parameters())查看参数名
            params_1x = [param for name, param in net.named_parameters()
                if name not in ["classifier.3.weight", "classifier.3.bias"]]
            trainer = torch.optim.SGD([{'params': params_1x},
                                    {'params': net.classifier[3].parameters(),
                                        'lr': learning_rate * 10}],
                                    lr=learning_rate, weight_decay=0.001)
        else:
            trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                    weight_decay=0.001)
        train(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices,tqdm_open=False,plot_training=False,wandb_open=True)
    
    #开始训练
    train_fine_tuning(mobilenet_v3_small, 5e-5)
