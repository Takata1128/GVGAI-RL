from torch.optim import SGD, Adam, RMSprop


def get_optimizer(optimizer_name: str):
    if optimizer_name == 'SGD':
        return SGD
    elif optimizer_name == 'Adam':
        return Adam
    elif optimizer_name == 'RMSProp':
        return RMSprop
    else:
        raise NotImplementedError(
            f"Optimizer {optimizer_name} is not implemented!")
