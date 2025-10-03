import torch 
from einops import rearrange, einsum, reduce, repeat
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
        
class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device = None):
        '''
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        factory_kwargs = {"device":device}
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 构建角度矩阵
        pos_id = torch.tensor(list(range(max_seq_len)), **factory_kwargs)
        d_id = torch.tensor([theta ** (-2*k/d_k) for k in range(d_k//2)], **factory_kwargs)
        mat = einsum(pos_id, d_id, "max_seq_len, half_d_k -> max_seq_len half_d_k")
        
        # 构建旋转矩阵
        cos_m = torch.cos(mat)
        sin_m = torch.sin(mat)
        combined = torch.stack([cos_m, -sin_m, sin_m, cos_m], dim = -1)
        trans = rearrange(combined, "... (h w) -> ... h w", h=2, w=2)
        
        # 转为buffer
        self.register_buffer("trans", trans, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        if token_positions == None:
            s = x.shape[-2]
            token_positions = torch.arange(s).expand(*x.shape[:-2], s)
        trans = self.trans[token_positions] # 索引出对应的旋转矩阵
        pair_wise_x = rearrange(x, "... (half_d_k two) -> ... half_d_k two", half_d_k = self.d_k//2, two = 2)
        res = einsum(trans, pair_wise_x, "... half_d_k r c, ... half_d_k c -> ... half_d_k r")
        res = rearrange(res, "... half_d_k r -> ... (half_d_k r)")
        return res
    
class SafeSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        max_val = reduce(x, "... d -> ... 1", "max")
        safe_exp_x = torch.exp(x-max_val)
        sum_val = reduce(safe_exp_x, "... d -> ... 1", "sum")
        res = safe_exp_x / sum_val
        return res
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask:torch.Tensor | None = None):
        sqrt_d = torch.sqrt(torch.tensor(K.shape[-1], dtype=float, device=K.device))
        softmax = SafeSoftmax()
        QK = einsum(Q,K,"... seq_q d, ... seq_k d -> ... seq_q seq_k") / sqrt_d
        if mask != None:
            add = torch.where(mask, 0.0, -torch.inf)
            QK = QK + add
        soft_QK = softmax(QK)
        res = einsum(soft_QK, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
        return res
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_input:int, num_heads:int, d_qk:int, d_v:int, d_output:int, 
                 rope:RoPE|None = None, device=None, dtype=None):
        '''
        d_input: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_qk: int Totol dimensionality of Q & K
        d_v: int Total dimensionality of V
        d_output: int Dimensionality of the Transformer block outputs.
        rope: RoPE a given rope module
        '''
        factory_kwargs = {"device":device, "dtype":dtype}
        super().__init__()
        self.d_input = d_input
        self.num_heads = num_heads
        self.d_qk = d_qk
        self.d_v = d_v
        self.d_output = d_output
        if rope != None:
            self.rope = rope
        else:
            self.rope = None
        self.attention = ScaledDotProductAttention()
        
        self.wq = Linear(in_features=d_input, out_features=d_qk, **factory_kwargs)
        self.wk = Linear(in_features=d_input, out_features=d_qk, **factory_kwargs)
        self.wv = Linear(in_features=d_input, out_features=d_v, **factory_kwargs)
        self.wo = Linear(in_features=d_v, out_features=d_output, **factory_kwargs)
        
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor|None = None):
        '''
        x: [... seq d_input]
        '''
        seq = x.shape[-2]
        n_head_q = rearrange(self.wq(x), "... seq (num_head dh_qk) -> ... num_head seq dh_qk",
                             num_head=self.num_heads, dh_qk = self.d_qk // self.num_heads)
        n_head_k = rearrange(self.wk(x), "... seq (num_head dh_qk) -> ... num_head seq dh_qk",
                             num_head=self.num_heads, dh_qk = self.d_qk // self.num_heads)
        n_head_v = rearrange(self.wv(x), "... seq (num_head dh_v) -> ... num_head seq dh_v",
                             num_head=self.num_heads, dh_v = self.d_v // self.num_heads)
        if self.rope != None:
            n_head_q_rope = self.rope(n_head_q, token_positions)
            n_head_k_rope = self.rope(n_head_k, token_positions)
        else:
            n_head_q_rope = n_head_q
            n_head_k_rope = n_head_k
        
        mask = torch.tril(torch.ones(size=(seq,seq))).bool()
        n_head_atten_score = self.attention(n_head_q_rope,n_head_k_rope,n_head_v,mask)
        
        n_head_atten_score = rearrange(n_head_atten_score,
                                       "... num_head seq dh_v -> ... seq (num_head dh_v)")
        res = self.wo(n_head_atten_score)
        
        return res
        
class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int,
                 rope:RoPE|None = None, device=None, dtype=None):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        '''
        factory_kwargs = {"device":device, "dtype":dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.pre_attn_rmsnorm = RMSNorm(d_model=d_model, eps = 1e-5, **factory_kwargs)
        self.attention = MultiHeadSelfAttention(d_input=d_model,
                                                num_heads=num_heads,
                                                d_qk=d_model,
                                                d_v=d_model,
                                                d_output=d_model,
                                                rope=rope,
                                                **factory_kwargs)
        self.pre_ffn_rmsnorm = RMSNorm(d_model=d_model, eps = 1e-5, **factory_kwargs)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff,**factory_kwargs)
    
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.pre_attn_rmsnorm(x))
        x = x + self.ffn(self.pre_ffn_rmsnorm(x))
        return x
        
        
        
        
        
    

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
        
    @staticmethod
    def test_rope():
        print("\nRoPE sample:")
        x = torch.tensor([[3.,1.,2.,4.],[3.,2.,1.,4.]])
        print("x=", x)
        rope = RoPE(theta=torch.pi/10, d_k=4, max_seq_len=2)
        print("trans=", rope.trans)
        y = rope(x, torch.tensor([0,1]))
        print("y=", y)
        print("end\n")
    
    @staticmethod
    def test_safe_softmax():
        print("\nSafe-Softmax sample:")
        x = torch.tensor([[3.,1.,2.,4.],[3.,2.,1.,4.]])
        print("x=", x)
        softmax = SafeSoftmax()
        y = softmax(x)
        print("y=", y)
        print("end\n")
        
    @staticmethod
    def test_scaled_dot_product_attention():
        print("\nScaled Dot Product Attention sample:")
        Q = torch.tensor([[3.,1.,2.,4.],[3.,2.,1.,4.]])
        K = torch.tensor([[3.,1.,2.,4.],[3.,2.,1.,4.]])
        V = torch.tensor([[3.,1.,2.,4.],[3.,2.,1.,4.]])
        print("Q=", Q)
        print("K=", K)
        print("V=", V)
        attn = ScaledDotProductAttention()
        y = attn(Q,K,V)
        print("y=", y)
        print("end\n")
        
        

if __name__ == "__main__":
    Test_Modules.test_scaled_dot_product_attention()
        