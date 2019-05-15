import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

train_data = MNIST('mnist', download=True, train=True)
test_data = MNIST('mnist', download=True, train=False)


# https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

# size_out = 1/stride * (size_in + 2 * padding - dilatation * (kernel_size - 1) - 1) + 1

# in_channels (int) – Number of channels in the input image
# out_channels (int) – Number of channels produced by the convolution
# kernel_size (int or tuple) – Size of the convolving kernel
# stride (int or tuple, optional) – Stride of the convolution. Default: 1
# padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
# dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
# groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
# bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1)


class Batchfy(nn.Module):
    def forward(self, input):
        return input.view(-1, 10)


simple_cnn = nn.Sequential(nn.Conv2d(1, 8, (2, 2), stride=1),
                           nn.ReLU(),
                           nn.Conv2d(8, 32, (3, 3), stride=2, padding=1),
                           nn.ReLU(),
                           nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
                           nn.ReLU(),
                           Flatten(),
                           nn.Linear(1568, 10),
                           Batchfy()
                           )

converter = ToTensor()

img = converter(train_data[0][0])
img = img.view(-1, 1, 28, 28)

loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(simple_cnn.parameters())

# device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
device = torch.device('cpu')

simple_cnn = simple_cnn.to(device)

with torch.set_grad_enabled(True):

    simple_cnn.train()
    for epoch in range(5):
        total_loss = 0
        corrects = 0

        max_elements = 1000

        for i, (img, y) in enumerate(train_data):
            if i > max_elements:
                break

            img = converter(img).to(device)
            img = img.view(-1, 1, 28, 28)
            out = simple_cnn(img)
            y = y.view(1)
            loss = loss_function(out, y)

            optimiser.zero_grad()  # zero grad before propagating current grads
            loss.backward()
            optimiser.step()

            total_loss += loss.item()  # use item to get value and detach from computational graph

            corrects += 1 if y == torch.argmax(out) else 0
        print('running error', total_loss / max_elements, 'accuracy:', corrects / max_elements)
