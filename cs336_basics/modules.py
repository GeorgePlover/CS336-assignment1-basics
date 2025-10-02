import torch 
from einops import rearrange, einsum
import torch.nn as nn

class Parameter_Init:
    def __init__(self):
        pass
    
    @staticmethod
    def linear_weights(weight: nn.Parameter, d_in: int, d_out: int):
        sigma_square = 2.0 / (d_in + d_out)
        sigma = sigma_square ** 0.5
        nn.init.trunc_normal_(tensor=weight, mean=0, std=sigma,
                              a=-3*sigma, b=3*sigma)
    @staticmethod
    def embedding(weight: nn.Parameter):
        sigma_square = 1
        sigma = sigma_square ** 0.5
        nn.init.trunc_normal_(tensor=weight, mean=0, std=sigma,
                              a=-3*sigma, b=3*sigma)
        

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        '''
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        factory_kwargs = {"device": device, "dtype": dtype} # 简化代码，减少调用时重复显式赋值
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            data=torch.empty(size=(out_features, in_features), **factory_kwargs), # 注意权重矩阵维度反过来存，进行线性向量乘矩阵时，有助于访问连续内存
            requires_grad=True
        )
        Parameter_Init.linear_weights(self.weight, in_features, out_features)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_size, out_size in_size -> ... out_size")
    

    

class Test_Modules:
    def __init__(self):
        pass
    
    @staticmethod
    def test_linear():
        print("\nLinear sample:")
        x = torch.ones(2,3)
        print("x=", x)
        L = Linear(in_features=3, out_features=4)
        print("linear weight=", L.weight)
        y = L(x)
        print("y=", y)
        for key, value in L.state_dict().items():
            print(f"{key}: {value.shape}")
        print("end\n")
        
        

if __name__ == "__main__":
    Test_Modules.test_linear()
        