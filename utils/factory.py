from models.ilpc import ILPC
from models.loss import PGD_Prototype_Novel

def get_model(model_name, args):
    name = model_name.lower()
    if name == 'ilpc':
        return ILPC(args)
    else:
        assert 0

def get_loss(loss_name, args):
    name = loss_name.lower()
    if name == 'pgd_prototype_novel':
        return PGD_Prototype_Novel()
    else:
        assert 0
