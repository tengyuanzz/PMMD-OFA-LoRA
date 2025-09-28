import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.ld_pretrained_path = configs.ld_pretrained_path
        self.kernel_sizes = [configs.kernel_sizes * y for y in [5,10]]
        self.LD = RobDecomp(dim=configs.input_dim, kernel_sizes=self.kernel_sizes, steps=configs.steps).to(device)

        if self.ld_pretrained_path is not None:
            state = torch.load(self.ld_pretrained_path, map_location='cpu')
            state = {k.replace('module.', ''): v for k, v in state.items()}
            self.LD.load_state_dict(state, strict=False)
            if configs.freeze_decomp:
                for p in self.LD.parameters():
                    p.requires_grad = False

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        # Linear model for trend target prediction
        self.trend_linear = nn.Linear(configs.seq_len * configs.input_dim, configs.pred_len).to(device)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                # print('name and param', name)
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0



    def forward(self, x, itr):
        trend, seasonal = self.LD(x)
        B, L, M = seasonal.shape

        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        seasonal = seasonal - means
        seasonal /= stdev
        trend = trend - means
        trend /= stdev

        seasonal = rearrange(seasonal, 'b l m -> b m l')
        seasonal = self.padding_patch_layer(seasonal)
        seasonal = seasonal.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        seasonal = rearrange(seasonal, 'b m n p -> (b m) n p')

        outputs = self.in_layer(seasonal)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        # Process trend component with linear model
        trend_flat = trend.reshape(B, -1)  # Flatten trend to [B, L*M]
        trend_pred = self.trend_linear(trend_flat)  # Predict trend for target variable
        trend_pred = trend_pred.unsqueeze(-1)  # Add dimension for variables [B, pred_len, 1]
        
        # Add trend prediction to the target dimension of the seasonal output
        outputs[:, :, -1:] = outputs[:, :, -1:] + trend_pred

        outputs = outputs * stdev + means

        return outputs + trend


class RobDecomp(nn.Module):
    """
    Robust Series Decomposition block.
    Implements Algorithm 1 from “Robformer”:
      1) Learn λ = [λ1, λ2] for mix of 1st- and 2nd-order moving averages.
      2) For each kernel size τ_k, compute X_t^{(1)} and X_t^{(2)} via avg pooling.
      3) Mix them: X_t,τ_k = λ1 * X_t^{(1)} + λ2 * X_t^{(2)}.
      4) Learn ω τ over the K different τ_k and mix to get final trend X_t.
      5) Seasonal part X_s = X − X_t.
    """
    def __init__(self, dim, kernel_sizes, hidden_dim=None, steps=3):
        """
        Args:
          dim           : number of features d
          kernel_sizes  : list of integers [τ1, τ2, …, τK]
          hidden_dim    : internal dimension for weight MLPs (default = dim)
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes
        K = len(kernel_sizes)
        hidden_dim = hidden_dim or dim
        self.steps = steps

        # map global summary → 2 weights λ1,λ2
        self.lambda_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        # map global summary → K weights ω₁…ω_K
        self.omega_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, x):
        B, L, d = x.shape
        residual = x.clone()
        trend_total = torch.zeros_like(x)
        
        for step in range(self.steps):
            summary = residual.mean(dim=1)
            lam = F.softmax(self.lambda_mlp(summary), dim=-1)  # (B, 2)
            λ1 = lam[:, 0].view(B, 1, 1)
            λ2 = lam[:, 1].view(B, 1, 1)

            trend_components = []
            for τ in self.kernel_sizes:
                x_perm = residual.permute(0, 2, 1)
                left = (τ - 1) // 2
                right = (τ - 1) - left
                x1 = F.avg_pool1d(F.pad(x_perm, (left, right), mode='reflect'),
                                   kernel_size=τ, stride=1)
                x2 = F.avg_pool1d(F.pad(x1, (left, right), mode='reflect'),
                                   kernel_size=τ, stride=1)
                x1 = x1.permute(0, 2, 1)
                x2 = x2.permute(0, 2, 1)
                t_k = λ1 * x1 + λ2 * x2
                trend_components.append(t_k)

            trend_stack = torch.stack(trend_components, dim=1)
            omega = F.softmax(self.omega_mlp(summary), dim=-1).view(B, len(self.kernel_sizes), 1, 1)
            x_trend_step = (trend_stack * omega).sum(dim=1)

            # Accumulate trend progressively
            trend_total += x_trend_step
            residual = residual - x_trend_step

        x_seasonal = residual
        return trend_total, x_seasonal
