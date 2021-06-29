from torch import nn
from .local_loss_blocks import LocalLossBlock
from utils.models import ViewLayer


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

    def lr_scheduler_step(self):
        for layer in self.iter_local_loss_layers():
            layer.lr_scheduler_step()

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

    def get_base_inference_layers(self):

        """layers = []
        for name, module in self._modules.items():
            try:
                layers + [(f"{name}." + x, y) for x, y in module.get_base_inference_layers()]
            except:
                if not isinstance(module, ViewLayer):
                    layers.append((module))"""

        blocked_modules = []
        layers = []

        #print(len(list(self.named_modules())))

        for named_module in self.named_modules():
            module = named_module[1]
            if isinstance(module, LocalLossBlock):
                blocked_modules += [m[1] for m in module.named_modules()][2:]

            if len(list(module.children())) == 0 and not isinstance(module, ViewLayer) and module not in blocked_modules:
                layers.append(named_module)

        return layers
