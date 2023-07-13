#!/usr/bin/python3
# export TORCH_BLADE_DEBUG_LOG=true 

import torch
import torch_blade


class QLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, org_dtype, bit=8):
        super().__init__()
        self.org_dtype = org_dtype
        self.q_min, self.q_max = -1 << bit - 1, (1 << bit - 1) - 1  # int8
        self.register_buffer(
            'weight', torch.zeros((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer(
            'scale_w', torch.zeros((out_features), dtype=torch.float32)
        )
        self.register_buffer('bias', torch.zeros(out_features, dtype=org_dtype))

    @torch.no_grad()
    def quantize(self, x, dim=-1):
        x_max = torch.max(torch.abs(x), dim, keepdim=True)[0].float()
        scale = x_max / self.q_max
        q_x = torch.round(x / scale)
        q_x = torch.clamp(q_x, min=self.q_min, max=self.q_max)
        q_x = q_x.to(torch.int8)
        return q_x, scale

    @torch.no_grad()
    def fake_int8_gemm(self, q_x, scale_x):
        q_x = q_x.float()
        q_w = self.weight.float()
        q_y = torch.functional.F.linear(q_x, q_w) # q_x * q_w
        q_y = q_y * scale_x * self.scale_w #
        q_y = q_y.to(self.org_dtype) #
        return q_y

    @torch.no_grad()
    def forward(self, x):
        q_x, scale_x = self.quantize(x) #
        q_y = self.fake_int8_gemm(q_x, scale_x) + self.bias
        return q_y

    @classmethod
    def from_float(cls, module):
        q_module = cls(
            module.weight.shape[1],
            module.weight.shape[0],
            org_dtype=module.weight.dtype,
        )
        q_module.weight, q_scale = q_module.quantize(module.weight)
        q_module.scale_w = q_scale.reshape(-1)
        q_module.bias = module.bias
        return q_module


# For 13b-8k model
class QParallelLinear(QLinear):
    def __init__(
        self, in_features, out_features, org_dtype, skip_bias_add=False, bit=8
    ):
        super().__init__(in_features, out_features, org_dtype, bit)
        self.skip_bias_add = skip_bias_add

    @torch.no_grad()
    def forward(self, x):
        q_x, scale_x = self.quantize(x)
        q_y = self.fake_int8_gemm(q_x, scale_x)
        if self.skip_bias_add:
            return q_y, self.bias
        q_y = q_y + self.bias
        return q_y, None

    @classmethod
    def from_float(cls, module):
        q_module = cls(
            module.weight.shape[1],
            module.weight.shape[0],
            org_dtype=module.weight.dtype,
            skip_bias_add=module.skip_bias_add,
        )
        q_module.weight, q_scale = q_module.quantize(module.weight)
        q_module.scale_w = q_scale.reshape(-1)
        q_module.bias = module.bias
        return q_module


# For 13b-8k model
def do_quant(model):
    func = QParallelLinear.from_float
    from megatron.model.transformer import ParallelAttention, ParallelMLP

    for name, m in model.named_modules():
        print('Quantize {}'.format(name))
        if isinstance(m, ParallelAttention):
            m.query_key_value = func(m.query_key_value)
            m.dense = func(m.dense)
        elif isinstance(m, ParallelMLP):
            m.w1 = func(m.w1)
            m.w2 = func(m.w2)
            m.dense_4h_to_h = func(m.dense_4h_to_h)


if __name__ == '__main__':
    input_data = torch.randn((4, 32), dtype=torch.float32)
    fp32_model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.Linear(64, 8),
    )
    int8_model = torch.nn.Sequential(
        QLinear.from_float(fp32_model[0]),
        QLinear.from_float(fp32_model[1]),
    )
    print(fp32_model(input_data))
    print(int8_model(input_data))