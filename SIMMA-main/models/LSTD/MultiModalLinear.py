import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad on both ends of time series to maintain length.
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block.
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiModalLinear(nn.Module):
    """
    Decomposition-Linear model.
    """

    def __init__(self, args=None):
        super(MultiModalLinear, self).__init__()
        self.args = args
        self.logit_linear = nn.Linear(self.args.in_len, 1)

        kernel_size = args.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Space_moving = nn.ModuleList()
        self.Linear_Space_trend = nn.ModuleList()
        self.Linear_Time_moving = nn.ModuleList()
        self.Linear_Time_trend = nn.ModuleList()

        # LSTD for each modality
        for i in range(self.args.num_modals):
            self.Linear_Time_moving.append(nn.Linear(self.args.in_len, self.args.in_len, bias=True))
            self.Linear_Time_trend.append(nn.Linear(self.args.in_len, self.args.in_len, bias=True))
            self.Linear_Space_moving.append(nn.Linear(self.args.input_dims, self.args.input_dims, bias=True))
            self.Linear_Space_trend.append(nn.Linear(self.args.input_dims, self.args.input_dims, bias=True))

    def forward(self, x):
        _x = x
        # x: [Batch, Input length, Modal_Channels]
        time_moving_init, time_trend_init = self.decompsition(x)
        # time_moving_init, time_trend_init: [Batch, Modal_Channels, Input length]
        time_moving_init, time_trend_init = time_moving_init.permute(0, 2, 1), time_trend_init.permute(0, 2, 1)
        # space_moving_init, space_trend_init: [Batch, Input length, Modal_Channels]
        space_moving_init, space_trend_init = self.decompsition(x.permute(0, 2, 1))
        space_moving_init, space_trend_init = space_moving_init.permute(0, 2, 1), space_trend_init.permute(0, 2, 1)

        time_moving_output = torch.zeros_like(time_moving_init).to(time_moving_init.device)
        time_trend_output = torch.zeros_like(time_trend_init).to(time_trend_init.device)
        space_moving_output = torch.zeros_like(space_moving_init).to(space_moving_init.device)
        space_trend_output = torch.zeros_like(space_trend_init).to(space_trend_init.device)

        # LSTD for each modality
        for i in range(self.args.num_modals):
            time_moving_output += self.Linear_Time_moving[i](time_moving_init)
            time_trend_output += self.Linear_Time_trend[i](time_trend_init)
            space_moving_output += self.Linear_Space_moving[i](space_moving_init)
            space_trend_output += self.Linear_Space_trend[i](space_trend_init)

        # time_features, space_features: [Batch, Output length, Modal_Channels]
        time_features = (time_moving_output + time_trend_output).permute(0, 2, 1)
        space_features = (space_moving_output + space_trend_output)
        x_time_space = time_features + space_features + _x

        if self.args.use_align or self.args.use_transformer:
            return x_time_space

        logit = self.logit_linear(x_time_space.mean(dim=2))
        return logit
