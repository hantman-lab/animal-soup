import torch


class L2(torch.nn.Module):
    """
    L2 regularization.
    """
    def __init__(self, model: torch.nn.Module, alpha: float):
        super().__init__()

        self.alpha = alpha

        to_decay = list()

        for name, param in model.named_parameters():
            if (not param.requires_grad) or \
                    ('batchnorm' in name.lower()) or \
                    ('bn' in name.lower()) or\
                    ('bias' in name.lower()) or\
                    (param.ndim == 1):
                continue
            else:
                to_decay.append(name)

        self.keys = to_decay

    def forward(self, model):
        # https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # note that soumith's answer is wrong because it uses W.norm, which takes the square root
        l2_loss = 0 # torch.tensor(0., requires_grad=True)
        for key, param in model.named_parameters():
            if key in self.keys:
                l2_loss += param.pow(2).sum( ) * 0.5

        return l2_loss * self.alpha

