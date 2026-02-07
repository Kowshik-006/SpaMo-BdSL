import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveNorm1d(nn.Module):
    """
    Normalization layer that uses BatchNorm1d when possible (temporal dim > 1),
    but falls back to LayerNorm when temporal dimension is 1 to avoid errors.
    This preserves BatchNorm behavior in most cases while handling edge cases.
    """
    def __init__(self, num_features):
        super(AdaptiveNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.ln = nn.LayerNorm(num_features)
        self.num_features = num_features
    
    def forward(self, x):
        # x shape: (batch, channels, temporal)
        if x.shape[2] > 1:
            # Use BatchNorm when we have multiple temporal positions
            return self.bn(x)
        else:
            # Use LayerNorm when temporal dim is 1
            # LayerNorm expects (batch, seq_len, features), so permute
            x_permuted = x.permute(0, 2, 1)  # (batch, temporal, channels)
            normalized = self.ln(x_permuted)
            return normalized.permute(0, 2, 1)  # back to (batch, channels, temporal)


class SafeMaxPool1d(nn.Module):
    """
    MaxPool1d wrapper that handles cases where the input temporal dimension
    is too small for the pooling operation. If the input is too small, it uses
    adaptive pooling to ensure at least size 1 output.
    """
    def __init__(self, kernel_size, ceil_mode=False):
        super(SafeMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.ceil_mode = ceil_mode
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, ceil_mode=ceil_mode)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        # x shape: (batch, channels, temporal)
        temporal_dim = x.shape[2]
        
        # If input is too small for regular pooling, use adaptive pooling
        if temporal_dim < self.kernel_size:
            # Use adaptive pooling to get at least size 1 output
            return self.adaptive_pool(x)
        else:
            # Use regular pooling
            return self.pool(x)


class SafeConv1d(nn.Module):
    """
    Conv1d wrapper that handles cases where the input temporal dimension
    is too small for the convolution operation. If the input is too small,
    it pads the input to match the kernel size before applying convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SafeConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        # x shape: (batch, channels, temporal)
        temporal_dim = x.shape[2]
        
        # If input is too small for convolution, pad it
        if temporal_dim < self.kernel_size:
            # Calculate padding needed
            pad_size = self.kernel_size - temporal_dim
            # Pad on the right side (end of sequence)
            x = F.pad(x, (0, pad_size), mode='replicate')
        
        return self.conv(x)


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                # Use SafeMaxPool1d to handle small temporal dimensions
                modules.append(SafeMaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                # Use SafeConv1d to handle small temporal dimensions
                modules.append(
                    SafeConv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                    #MultiScale_TemporalConv(input_sz, self.hidden_size)
                )
                # Use adaptive normalization that preserves BatchNorm when possible
                modules.append(AdaptiveNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                # For pooling, divide by kernel size, but ensure at least 1
                # Use floor division to match the pooling behavior
                feat_len = feat_len // int(ks[1])
                feat_len = torch.clamp(feat_len, min=1)
            else:
                # For convolution, subtract (kernel_size - 1), but ensure at least 1
                feat_len = feat_len - (int(ks[1]) - 1)
                feat_len = torch.clamp(feat_len, min=1)
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,
            "feat_len": lgt.cpu(),
        }
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, stride=1)
        # Use adaptive normalization that preserves BatchNorm when possible
        self.bn1 = AdaptiveNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + residual  # Element-wise addition
        out = self.relu(out)
        return out


class GlorTemporalConv(nn.Module):
    def __init__(self, input_channels, output_channels, dilation_rate=1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)
        )
        self.layers.append(ResidualBlock(output_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 2, 1)

