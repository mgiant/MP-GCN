from .nets import MPGCN

def create(**kwargs):
    return MPGCN(**kwargs)
