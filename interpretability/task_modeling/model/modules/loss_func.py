import torch.nn as nn
import torch
class LossFunc():
    def __init__():
        pass
    def __call__(self, pred, target, act = None):
        pass

class RandomTargetLoss(LossFunc):
    def __init__(self, position_loss, pos_weight, act_weight):
        self.position_loss = position_loss
        self.pos_weight = pos_weight
        self.act_weight = act_weight
    def __call__(self, pred, target, act):
        pos_loss = self.pos_weight * self.position_loss(pred, target)
        act_loss = self.act_weight * nn.MSELoss()(act, torch.zeros_like(act))
        return pos_loss + act_loss
    
class L1LossFunc(LossFunc):
    def __init__(self):
        pass
    def __call__(self, pred, target, act):
        return nn.L1Loss()(pred, target)