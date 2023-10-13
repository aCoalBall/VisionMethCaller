import torch
from torch import nn


def swish(x):
    """Swish activation

    Swish is self-gated linear activation :math:`x sigma(x)`

    For details see: https://arxiv.org/abs/1710.05941

    Note:
        Original definition has a scaling parameter for the gating value,
        making it a generalisation of the (logistic approximation to) the GELU.
        Evidence presented, e.g. https://arxiv.org/abs/1908.08681 that swish-1
        performs comparable to tuning the parameter.

    """
    return x * torch.sigmoid(x)


class RemoraNet(nn.Module):
    _variable_width_possible = False

    def __init__(
        self,
        size=64,
        #kmer_len=9,
        num_out=2,
    ):
        super().__init__()
        self.sig_conv1 = nn.Conv1d(1, 4, 5)
        self.sig_bn1 = nn.BatchNorm1d(4)
        self.sig_conv2 = nn.Conv1d(4, 16, 5)
        self.sig_bn2 = nn.BatchNorm1d(16)
        self.sig_conv3 = nn.Conv1d(16, size, 9, 3)
        self.sig_bn3 = nn.BatchNorm1d(size)

        self.seq_conv1 = nn.Conv1d(1, 16, 3, 1)
        self.seq_bn1 = nn.BatchNorm1d(16)
        self.seq_conv2 = nn.Conv1d(16, size, 3, 1)
        self.seq_bn2 = nn.BatchNorm1d(size)

        self.merge_conv1 = nn.Conv1d(size, size, 5)
        self.merge_bn = nn.BatchNorm1d(size)
        self.lstm1 = nn.LSTM(size, size, 1)
        self.lstm2 = nn.LSTM(size, size, 1)

        self.fc = nn.Linear(size, num_out)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sigs, seqs):
        # inputs are BFT (batch, feature, time)
        sigs_x = swish(self.sig_bn1(self.sig_conv1(sigs)))
        sigs_x = swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = swish(self.seq_bn2(self.seq_conv2(seqs_x)))

        z = torch.cat((sigs_x, seqs_x), 2)

        z = swish(self.merge_bn(self.merge_conv1(z)))
        z = z.permute(2, 0, 1)
        z = swish(self.lstm1(z)[0])
        z = torch.flip(swish(self.lstm2(torch.flip(z, (0,)))[0]), (0,))
        z = z[-1].permute(0, 1)

        z = self.fc(z)

        return z
