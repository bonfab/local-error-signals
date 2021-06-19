from torch import nn
from .local_loss_blocks import LocalLossBlock


class LocalLossNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.local_eval = False

    def local_loss_eval(self):
        self.local_eval = True
        for ll_layer in self.iter_local_loss_layers():
            ll_layer.local_loss_eval()

    def local_loss_train(self):
        self.local_eval = False
        for ll_layer in self.iter_local_loss_layers():
            ll_layer.local_loss_train()

    def parameters(self):
        if not self.args.backprop:
            return self.layers[-1].parameters()
        else:
            return super(LocalLossNet, self).parameters()

    def iter_local_loss_layers(self):
        for layer in self.layers:
            if isinstance(layer, LocalLossBlock) or isinstance(layer, LocalLossNet):
                yield layer

    def set_learning_rate(self, lr):
        for layer in self.iter_local_loss_layers():
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for layer in self.iter_local_loss_layers():
            layer.optim_zero_grad()

    def optim_step(self):
        for layer in self.iter_local_loss_layers():
            layer.optim_step()

    def forward(self, x, y=None, y_onehot=None):

        assert not (y is None or y_onehot is None) or self.local_eval

        if not self.local_eval:
            total_loss = 0
            for layer in self.layers:
                if isinstance(layer, LocalLossBlock) or isinstance(layer, LocalLossNet):
                    x, loss = layer(x, y, y_onehot)
                    total_loss += loss
                else:
                    x = layer(x)
                #print(x.shape)

            return x, total_loss
        else:
            for layer in self.layers:
                x = layer(x)
            return x
