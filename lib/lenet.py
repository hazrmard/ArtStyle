import torch.nn.functional as F
import torch.nn as nn

from lib.fully_connected_net import FullyConnectedNet


class LeNet(nn.Module):
    def __init__(self, input_size,
                       output_size,
                       batch_norm,

                       use_pooling,
                       pooling_method,

                       conv1_kernel_size,
                       conv1_num_kernels,
                       conv1_stride,
                       conv1_dropout,

                       pool1_kernel_size,
                       pool1_stride,

                       conv2_kernel_size,
                       conv2_num_kernels,
                       conv2_stride,
                       conv2_dropout,

                       pool2_kernel_size,
                       pool2_stride,

                       fcs_hidden_size,
                       fcs_num_hidden_layers,
                       fcs_dropout):

        # print('LeNet.__init__: conv2_stride =', conv2_stride)
        super(LeNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = fcs_hidden_size
        self.output_size = output_size

        self.batch_norm = batch_norm

        input_channel = 3

        # If not using pooling, set all pooling operations to 1 by 1.
        if use_pooling == False:
            pool1_kernel_size = 1
            pool1_stride = 1
            pool2_kernel_size = 1
            pool2_stride = 1


        # Conv1
        conv1_output_size = (conv1_num_kernels,
                            (input_size[0] - conv1_kernel_size) / conv1_stride + 1,
                            (input_size[1] - conv1_kernel_size) / conv1_stride + 1)

        self.conv1 = nn.Conv2d(input_channel, conv1_num_kernels, kernel_size=conv1_kernel_size, stride=[conv1_stride]) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        nn.init.kaiming_normal_(self.conv1.weight.data)
        self.conv1.bias.data.fill_(0)

        self.conv1_drop = nn.Dropout2d(p=conv1_dropout)
        if self.batch_norm is True:
            self.batch_norm1 = nn.BatchNorm2d(conv1_num_kernels)

        # Pool1
        pool1_output_size = (conv1_num_kernels,
                            (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1,
                            (conv1_output_size[2] - pool1_kernel_size) / pool1_stride + 1)

        self.pool1 = nn.MaxPool2d(pool1_kernel_size, stride=pool1_stride) # stride=pool1_kernel_size by default


        # Conv2
        conv2_output_size = (conv2_num_kernels,
                            (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1,
                            (pool1_output_size[2] - conv2_kernel_size) / conv2_stride + 1)

        # print('\n\nbefore conv2. conv1_num_kernels =', conv1_num_kernels, '; conv2_num_kernels =', conv2_num_kernels, '; conv2_kernel_size =', conv2_kernel_size, '; conv2_stride =', conv2_stride)
        self.conv2 = nn.Conv2d(conv1_num_kernels, conv2_num_kernels, kernel_size=conv2_kernel_size, stride=conv2_stride) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        # print('\n\nafter conv2')
        nn.init.kaiming_normal_(self.conv2.weight.data)
        self.conv2.bias.data.fill_(0)

        self.conv2_drop = nn.Dropout2d(p=conv2_dropout)
        if self.batch_norm is True:
            self.batch_norm2 = nn.BatchNorm2d(conv2_num_kernels)

        # Pool2
        pool2_output_size = (conv2_num_kernels,
                            (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1,
                            (conv2_output_size[2] - pool2_kernel_size) / pool2_stride + 1)

        self.pool2 = nn.MaxPool2d(pool2_kernel_size, stride=pool2_stride) # stride=pool1_kernel_size by default


        # FCs
        fcs_input_size = pool2_output_size[0] * pool2_output_size[1] * pool2_output_size[2]
        # print('lenet: fcs_input_size = %s * %s * %s = %s' % (pool2_output_size[0],pool2_output_size[1], pool2_output_size[2], fcs_input_size))
        self.fcs = FullyConnectedNet(fcs_input_size,
                                     output_size,

                                     fcs_dropout,
                                     batch_norm,

                                     fcs_hidden_size,
                                     fcs_num_hidden_layers)


    def forward(self, x):
        # pytorch.conv1d accepts shape (Batch, Channel, Width)
        # pytorch.conv2d accepts shape (Batch, Channel, Height, Width)
        # import code; code.interact(local=dict(globals(), **locals()))
        # num_elements = int(x.shape[1] / 2)
        # x = x.view(-1, 2, num_elements)
        # BUG: Currently no batches
        # print('forward: input: x.size() =', x.size())
        x = self.conv1(x)
        # print('forward: after conv1: x.size() =', x.size())
        x = self.pool1(x)
        # print('forward: after pool1: x.size() =', x.size())
        x = F.relu(x)
        # print('forward: after relu: x.size() =', x.size())
        x = self.conv1_drop(x)
        # print('forward: after conv1_drop: x.size() =', x.size())
        if self.batch_norm is True:
            x = self.batch_norm1(x)
            # print('forward: after batch_norm1: x.size() =', x.size())

        x = self.conv2(x)
        # print('forward: after conv2: x.size() =', x.size())
        x = self.pool2(x)
        # print('forward: after pool2: x.size() =', x.size())
        x = F.relu(x)
        # print('forward: after relu: x.size() =', x.size())
        x = self.conv2_drop(x)
        # print('forward: after conv2_drop: x.size() =', x.size())
        if self.batch_norm is True:
            x = self.batch_norm2(x)
            # print('forward: after batch_norm2: x.size() =', x.size())

        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        # print('forward: after flatten: x.size() =', x.size())

        x = self.fcs.forward(x)
        # print('forward: after fcs: x.size() =', x.size())

        return F.log_softmax(x, dim=1)
