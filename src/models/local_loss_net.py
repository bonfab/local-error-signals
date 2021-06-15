from torch import nn
from .local_loss_blocks import LocalLossBlock


class LocalLossNet(nn.Module):

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

    def forward(self, x, y, y_onehot):
        total_loss = 0
        for layer in self.layers:
            if isinstance(layer, LocalLossBlock) or isinstance(layer, LocalLossNet):
                x, loss = layer(x, y, y_onehot)
                total_loss += loss
            else:
                x = layer(x)
            #print(x.shape)

        return x, total_loss
