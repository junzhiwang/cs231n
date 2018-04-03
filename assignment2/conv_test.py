import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


NUM_TRAIN = 49000
NUM_VAL = 1000

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=T.ToTensor())
loader_train = DataLoader(
    cifar10_train,
    batch_size=64,
    sampler=ChunkSampler(
        NUM_TRAIN,
        0))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(
    cifar10_val,
    batch_size=64,
    sampler=ChunkSampler(
        NUM_VAL,
        NUM_TRAIN))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                            transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)


dtype = torch.FloatTensor  # the CPU datatype

# Constant to control how frequently we print train loss
print_every = 100

# This is a little utility that we'll use to reset the model
# if we want to re-initialize all our parameters


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)


class MyThreeLayerConvNet(nn.Module):
    def __init__(self):
        super(MyThreeLayerConvNet, self).__init__()
        """
        [Conv - Relu - Pooling] * 3 - Affine
        3*32*32 - 32*32*32 - 32*16*16 - 32*8*8 - Affine - Relu - Affine -> pred
        """
        self.in_channel = [3, 128, 128]
        self.out_channel = [128, 128, 64]
        self.kernel_size = [3, 3, 3]
        self.padding = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.pool_size = [2, 2, 2]
        self.pool_stride = [2, 2, 2]
        self.affine_in = [1024, 256]
        self.affine_out = [256, 10]
        self.layers = {}

        for i in range(3):
            self.layers['conv{0}'.format(i)] = nn.Conv2d(in_channels=self.in_channel[i], out_channels=self.out_channel[i],
                kernel_size=self.kernel_size[i], padding=self.padding[i], stride=self.stride[i])
            self.layers['relu{0}'.format(i)] = nn.ReLU()
            self.layers['pool{0}'.format(i)] = nn.MaxPool2d(kernel_size=self.pool_size[i])

        for i in range(2):
            self.layers['affine{0}'.format(i)] = nn.Linear(self.affine_in[i], self.affine_out[i])

        # Store parameters to let pytorch track
        # https://discuss.pytorch.org/t/error-optimizer-got-an-empty-parameter-list/1501
        # https://discuss.pytorch.org/t/valueerror-optimizer-got-an-empty-parameter-list-for-resnet18-project-with-extra-upsampling-layers/9104/2

        layers = []
        for i in range(3):
            layers.append(self.layers['conv{0}'.format(i)])
            layers.append(self.layers['relu{0}'.format(i)])
            layers.append(self.layers['pool{0}'.format(i)])
        layers.append(Flatten())
        for i in range(2):
            layers.append(self.layers['affine{0}'.format(i)])
            if i == 0:
                layers.append(self.layers['relu{0}'.format(i)])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


myConvNet = MyThreeLayerConvNet()


loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.SGD(myConvNet.parameters(), lr=1.5e-2, momentum=0.9, weight_decay=0.001)

myConvNet.train()


def train(model, loss_fn, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


# torch.cuda.random.manual_seed(12345)
# fixed_model_gpu.apply(reset)
train(myConvNet, loss_fn, optimizer, num_epochs=12)

check_accuracy(myConvNet, loader_val)

"""
fixed_model.type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.RMSprop(fixed_model.parameters())

# Now we're going to feed a random batch into the model you defined and
# make sure the output is the right size
x = torch.randn(64, 3, 32, 32).type(dtype)
# Construct a PyTorch Variable out of your input data
x_var = Variable(x.type(dtype))
ans = fixed_model(x_var)        # Feed it through the model!

np.array_equal(np.array(ans.size()), np.array([64, 10]))

# This sets the model in "training" mode. This is relevant for some layers that may have different behavior
# in training mode vs testing mode, such as Dropout and BatchNorm.
fixed_model.train()

# Load one batch at a time.
for t, (x, y) in enumerate(loader_train):
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).long())

    # This is the forward pass: predict the scores for each class, for each x
    # in the batch.
    scores = fixed_model(x_var)

    # Use the correct y values and the predicted y values to compute the loss.
    loss = loss_fn(scores, y_var)

    if (t + 1) % print_every == 0:
        print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

    # Zero out all of the gradients for the variables which the optimizer will
    # update.
    optimizer.zero_grad()

    # This is the backwards pass: compute the gradient of the loss with respect to each
    # parameter of the model.
    loss.backward()

    # Actually update the parameters of the model using the gradients computed
    # by the backwards pass.
    optimizer.step()


def train(model, loss_fn, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


# torch.cuda.random.manual_seed(12345)
# fixed_model_gpu.apply(reset)
train(fixed_model, loss_fn, optimizer, num_epochs=1)
check_accuracy(fixed_model, loader_val)

"""
