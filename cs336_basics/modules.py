import torch 
from einops import rearrange, einsum, reduce
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
        
    @staticmethod
    def rmsnorm(gates: nn.Parameter):
        nn.init.constant_(tensor=gates, val=1.0)
        

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

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            data=torch.empty(size=(num_embeddings, embedding_dim), **factory_kwargs), # 让embedding维度连续存放
            requires_grad=True
        )
        Parameter_Init.embedding(self.weight)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.weight[x] # 此处直接调用tensor的索引算子；注意到下标是 [0,num_embeddings-1]
        
class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device = None, dtype = None):
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        factory_kwargs = {"device":device, "dtype":dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gates = nn.Parameter(
            data=torch.empty(size=(d_model,), **factory_kwargs),
            requires_grad=True
        )
        Parameter_Init.rmsnorm(self.gates)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        # trans to float32
        x_type = x.dtype
        x = x.to(torch.float32)
        
        # calc
        mean_square = reduce(x * x,"... d_model -> ... 1", 'mean') # 对应位置相乘求平均，保留维度，而不是压掉，以便后续 x/root 广播 
        root = torch.sqrt(mean_square + self.eps)
        res = einsum(x / root, self.gates, "... d_model, d_model -> ... d_model")
        
        # trans back
        return res.to(x_type)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        '''
        d_model:int embedding size
        d_ff:int intermediate size, is better to be 8/3 d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        SwiGLU(x) = W2 @ ( (W1@x) o-mul sigmoid(W1@x)  o-mul W3@x))
        W1,W3: [d_ff, d_model]
        W2: [d_model, d_ff]
        '''
        factory_kwargs = {"device":device, "dtype":dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff= d_ff
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)
    
    def forward(self, x: torch.Tensor):
        w1x = self.w1(x)
        w3x = self.w3(x)
        silu = w1x * torch.sigmoid(w1x)
        res = self.w2(silu * w3x)
        
        return res
        
    

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
    
    @staticmethod
    def test_embedding():
        print("\nEmbedding sample:")
        x = torch.tensor([[3,1,2],[3,2,1]])
        print("x=", x)
        E = Embedding(num_embeddings=4, embedding_dim=4)
        print("weight=", E.weight)
        y = E(x)
        print("y=", y)
        print("end\n")
        
    @staticmethod
    def test_rmsnorm():
        print("\nRMSNorm sample:")
        x = torch.tensor([[3.,1.,2.],[3.,2.,1.]])
        print("x=", x)
        rms = RMSNorm(d_model=x.shape[-1], eps=1e-5)
        print("gates=", rms.gates)
        y = rms(x)
        print("y=", y)
        print("end\n")
        
    @staticmethod
    def test_swiglu():
        print("\nSwiGLU sample:")
        x = torch.tensor([[3.,1.,2.],[3.,2.,1.]])
        print("x=", x)
        swiglu = SwiGLU(d_model=x.shape[-1], d_ff = 8 * x.shape[-1] // 3)
        print("w1=", swiglu.w1.weight)
        print("w2=", swiglu.w2.weight)
        print("w3=", swiglu.w3.weight)
        y = swiglu(x)
        print("y=", y)
        print("end\n")
        
        

if __name__ == "__main__":
    Test_Modules.test_swiglu()
        