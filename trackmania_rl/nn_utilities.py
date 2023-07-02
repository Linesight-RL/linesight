import torch
import math

def init_kaiming(layer, neg_slope=0, nonlinearity="leaky_relu"):
    torch.nn.init.kaiming_normal_(layer.weight, a=neg_slope, mode="fan_out", nonlinearity=nonlinearity)
    torch.nn.init.zeros_(layer.bias)


def init_xavier(layer, gain=1.0):
    torch.nn.init.xavier_normal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)


def init_uniform(layer, a, b):
    torch.nn.init.uniform_(layer.weight, a=a, b=b)
    torch.nn.init.zeros_(layer.bias)


def init_normal(layer, mean, std):
    torch.nn.init.normal_(layer.weight, mean=mean, std=std)
    torch.nn.init.zeros_(layer.bias)


# From https://github.com/pfnet/pfrl/blob/2ad3d51a7a971f3fe7f2711f024be11642990d61/pfrl/utils/copy_param.py#L37
def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_dict = target_link.state_dict()
    source_dict = source_link.state_dict()
    for k, target_value in target_dict.items():
        source_value = source_dict[k]
        if source_value.dtype in [torch.float32, torch.float64, torch.float16]:
            assert target_value.shape == source_value.shape
            target_value.mul_(1 - tau)
            target_value.add_(tau * source_value)
        else:
            # Scalar type
            # Some modules such as BN has scalar value `num_batches_tracked`
            target_dict[k] = source_value
            assert False, "Soft scalar update should not happen"


def custom_weight_decay(target_link, decay_factor):
    target_dict = target_link.state_dict()
    for k, target_value in target_dict.items():
        target_value.mul_(decay_factor)

def LR_From_Schedule(LR_Schedule, current_step):
    LR_Schedule = sorted(LR_Schedule,key=lambda p:p[0]) #Sort by step in case it was not defined in sorted order
    assert(LR_Schedule[0][0]==0)
    LR_schedule_end_index = next((idx for idx,p in enumerate(LR_Schedule) if p[0]>current_step),-1) #Returns -1 if none is found
    if LR_schedule_end_index==-1:
        return LR_Schedule[-1][1]
    else:
        assert(LR_schedule_end_index>=1)
        LR_schedule_end_step = LR_Schedule[LR_schedule_end_index][0]
        LR_Schedule_begin_step = LR_Schedule[LR_schedule_end_index-1][0]
        LR_annealing_period = LR_schedule_end_step-LR_Schedule_begin_step
        LR_end_value = LR_Schedule[LR_schedule_end_index][1]
        LR_begin_value = LR_Schedule[LR_schedule_end_index-1][1]
        LR_ratio = LR_begin_value/LR_end_value
        assert(LR_annealing_period>0)
        return LR_begin_value*math.exp(-math.log(LR_ratio)*(current_step-LR_Schedule_begin_step)/LR_annealing_period)
