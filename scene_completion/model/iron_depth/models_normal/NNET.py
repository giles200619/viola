import torch.nn as nn

from .submodules import EESNU


class NNET(nn.Module):
    def __init__(self, args):
        super(NNET, self).__init__()
        self.min_kappa = 0.01
        self.output_dim = 1
        self.output_type = 'G'

        if args.NNET_architecture == 'BN':
            self.n_net = EESNU(BN=True)
        else:
            self.n_net = EESNU(BN=False)

    def forward(self, img, **kwargs):
        return self.n_net(img, **kwargs)

