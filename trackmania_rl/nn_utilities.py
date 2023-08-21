import math

import torch


def init_kaiming(layer, neg_slope=0, nonlinearity="leaky_relu"):
    torch.nn.init.kaiming_normal_(layer.weight, a=neg_slope, mode="fan_out", nonlinearity=nonlinearity)
    torch.nn.init.zeros_(layer.bias)


def init_xavier(layer, gain=1.0):
    torch.nn.init.xavier_normal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)


def init_orthogonal(layer, gain=1.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.zeros_(layer.bias)


def init_uniform(layer, a, b):
    torch.nn.init.uniform_(layer.weight, a=a, b=b)
    torch.nn.init.zeros_(layer.bias)


def init_normal(layer, mean, std):
    torch.nn.init.normal_(layer.weight, mean=mean, std=std)
    torch.nn.init.zeros_(layer.bias)


def linear_combination(a, b, alpha):
    assert a.shape == b.shape
    a.mul_(1 - alpha)
    a.add_(alpha * b)
    return a


# From https://github.com/pfnet/pfrl/blob/2ad3d51a7a971f3fe7f2711f024be11642990d61/pfrl/utils/copy_param.py#L37
def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_dict = target_link.state_dict()
    source_dict = source_link.state_dict()
    for k, target_value in target_dict.items():
        source_value = source_dict[k]
        if source_value.dtype in [torch.float32, torch.float64, torch.float16]:
            target_value = linear_combination(target_value, source_value, tau)
        else:
            # Scalar type
            # Some modules such as BN has scalar value `num_batches_tracked`
            target_dict[k] = source_value
            assert False, "Soft scalar update should not happen"


def custom_weight_decay(target_link, decay_factor):
    target_dict = target_link.state_dict()
    for k, target_value in target_dict.items():
        target_value.mul_(decay_factor)


def from_exponential_schedule(schedule, current_step):
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    schedule_end_index = next((idx for idx, p in enumerate(schedule) if p[0] > current_step), -1)  # Returns -1 if none is found
    if schedule_end_index == -1:
        return schedule[-1][1]
    else:
        assert schedule_end_index >= 1
        schedule_end_step = schedule[schedule_end_index][0]
        schedule_begin_step = schedule[schedule_end_index - 1][0]
        annealing_period = schedule_end_step - schedule_begin_step
        end_value = schedule[schedule_end_index][1]
        begin_value = schedule[schedule_end_index - 1][1]
        ratio = begin_value / end_value
        assert annealing_period > 0
        return begin_value * math.exp(-math.log(ratio) * (current_step - schedule_begin_step) / annealing_period)


def from_linear_schedule(schedule, current_step):
    schedule = sorted(schedule, key=lambda p: p[0])  # Sort by step in case it was not defined in sorted order
    assert schedule[0][0] == 0
    schedule_end_index = next((idx for idx, p in enumerate(schedule) if p[0] > current_step), -1)  # Returns -1 if none is found
    if schedule_end_index == -1:
        return schedule[-1][1]
    else:
        assert schedule_end_index >= 1
        schedule_end_step = schedule[schedule_end_index][0]
        schedule_begin_step = schedule[schedule_end_index - 1][0]
        annealing_period = schedule_end_step - schedule_begin_step
        end_value = schedule[schedule_end_index][1]
        begin_value = schedule[schedule_end_index - 1][1]
        assert annealing_period > 0
        return begin_value + ((end_value - begin_value) * (current_step - schedule_begin_step)) / annealing_period
