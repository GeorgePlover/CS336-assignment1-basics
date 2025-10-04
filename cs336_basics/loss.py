from torch import Tensor
from torch import exp,log,arange,gather
from jaxtyping import Bool, Float, Int
from einops import reduce

def cross_entropy_loss(inputs: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]):
    # -log softmax() -> log sum(e^(v-max)) + max - v
    inputs_max = reduce(inputs, "... vocab_size -> ... 1", "max")
    inputs_sum = reduce(exp(inputs - inputs_max), "... vocab_size -> ... 1", "sum") 
    ans_1 = reduce(log(inputs_sum) + inputs_max, "... -> ", "mean")
    ans_2 =  reduce(gather(inputs, dim = -1, index = targets.unsqueeze(-1)), "... -> ", "mean")
    return ans_1 - ans_2