from . import MPGCN

__models = {
    'MPGCN': MPGCN,
}

def create(model_name, **kwargs):
    return __models[model_name].create(**kwargs)
