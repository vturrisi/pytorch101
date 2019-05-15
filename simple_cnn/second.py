import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets


train_data = MNIST('mnist', download=True, train=True)
test_data = MNIST('mnist', download=True, train=False)


# https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

# * size_out = 1/stride * (size_in + 2 * padding - dilatation * (kernel_size - 1) - 1) + 1 * #

# in_channels (int) – Number of channels in the input image

# out_channels (int) – Number of channels produced by the convolution

# kernel_size (int or tuple) – Size of the convolving kernel

# stride (int or tuple, optional) – Stride of the convolution. Default: 1

# padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0

# dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1

# groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1

# bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

# -------------------------------------------------------------------------------------

# https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d


# * computa a média e desvio padrão por canal sobre um minibatch de inputs 2D * #

# Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
# as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing
# Internal Covariate Shift .

# The mean and standard-deviation are calculated per-dimension over the mini-batches
# and \gammaγ and \betaβ are learnable parameter vectors of size C (where C is the input size).
# By default, the elements of \gammaγ are sampled from \mathcal{U}(0, 1)U(0,1)
# and the elements of \betaβ are set to 0.

# Also by default, during training this layer keeps running estimates of its
# computed mean and variance, which are then used for normalization during evaluation.
# The running estimates are kept with a default momentum of 0.1

# num_features – CC from an expected input of size (N, C, H, W)(N,C,H,W)

# eps – a value added to the denominator for numerical stability. Default: 1e-5

# momentum – the value used for the running_mean and running_var computation.
#   Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1

# affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True

# track_running_stats – a boolean value that when set to True, this module tracks the running mean and variance,
#   and when set to False, this module does not track such statistics and always uses
#   batch statistics in both training and eval modes. Default: True


class Flatten(nn.Module):
    def forward(self, input):
        batchsize = input.size(0)
        return input.view(batchsize, -1)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(nn.Conv2d(1, 8, (2, 2), stride=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 32, (3, 3), stride=2, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    Flatten(),
                                    nn.Linear(1568, 10),
                                    )

    def forward(self, imgs):
        out = self.layers(imgs)
        return out


class CustomMNISTLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.converter = ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, y = self.data[i]
        img = self.converter(img)
        img = img.view(-1, 1, 28, 28)
        y = y.view(1)
        return i, img, y


train_loader = CustomMNISTLoader(train_data)
test_loader = CustomMNISTLoader(test_data)

# device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
device = torch.device('cpu')
simple_cnn = SimpleCNN().to(device)


loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(simple_cnn.parameters())

with torch.set_grad_enabled(True):

    simple_cnn.train()
    for epoch in range(5):
        total_loss = 0
        corrects = 0

        max_elements = 100

        for i, img, y in train_loader:
            if i > max_elements:
                break

            img = img.to(device)

            out = simple_cnn(img)

            loss = loss_function(out, y)

            optimiser.zero_grad()  # zero grad before propagating current grads
            loss.backward()
            optimiser.step()

            total_loss += loss.item()  # use item to get value and detach from computational graph

            corrects += 1 if y == torch.argmax(out) else 0
        print('running error', total_loss / max_elements, 'accuracy:', corrects / max_elements)
