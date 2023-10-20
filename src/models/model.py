from torch import nn


class Xor(nn.Module):

    def __init__(self, hp):

        super().__init__()

        self.layer_1 = nn.Linear(hp.model.n_in, hp.model.n_hid)
        self.act_1 = nn.Sigmoid()
        self.layer_2 = nn.Linear(hp.model.n_hid, 1)
        self.act_2 = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act_1(x)
        x = self.layer_2(x)
        x = self.act_2(x)
        return x
