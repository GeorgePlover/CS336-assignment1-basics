# AGENTS-cn.md

## 项目概述
CS336 2025春季课程作业1：Transformer语言模型实现与BPE分词器训练。学生需要在`cs336_basics/`中实现核心组件，并通过`tests/adapters.py`将它们连接起来。

## 关键命令

### 环境与依赖
```bash
uv run <python_file>  # 自动解决依赖并激活环境
uv sync               # 安装/更新依赖
```

### 测试
```bash
uv run pytest                    # 运行所有测试
uv run pytest -v ./tests         # 详细测试输出
uv run pytest tests/test_tokenizer.py  # 运行单个测试文件
uv run pytest -k "test_name"     # 按名称运行特定测试
```

### 提交
```bash
./make_submission.sh  # 创建zip提交文件
```

## 架构说明

### 模块结构
- `cs336_basics/modules.py`: Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention模块
- `cs336_basics/tokenizer.py`: BPE分词器，包含训练和编码/解码功能
- `cs336_basics/loss.py`: 交叉熵损失实现
- `cs336_basics/pretokenization_example.py`: 用于并行预分词的块边界查找

### 测试适配器模式
所有实现必须通过`tests/adapters.py`连接。该文件从`cs336_basics`导入，并提供测试调用的包装函数。学生必须实现该文件中的`NotImplementedError`函数。

### 快照测试
测试使用快照测试（`tests/conftest.py`）：
- `numpy_snapshot` fixture用于NumPy数组（`.npz`文件在`tests/_snapshots/`中）
- `snapshot` fixture用于任意数据（`.pkl`文件）
- 使用`--update-snapshots`标志更新快照（仅限解决方案）

## 重要约定

### 张量类型注解
使用`jaxtyping`注解：
```python
from jaxtyping import Float, Int, Bool
from torch import Tensor

def func(x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
```

### 依赖项
- `einops`: 张量操作（`rearrange`, `einsum`, `reduce`, `repeat`）
- `jaxtyping`: 张量的类型注解
- `sortedcontainers`: 用于BPE分词器
- `regex`: 用于预分词模式的增强正则表达式

### 测试特点
- 测试直接从`cs336_basics`模块导入
- 部分测试使用`tests/fixtures/`中的fixture（GPT-2词汇表/合并）
- 存在内存限制的分词器测试（仅限Linux）
- 快照测试与参考实现进行比较

## 数据设置
在运行实验前下载所需数据（先检查数据是否存在，避免重复下载）：
```bash
mkdir -p data && cd data

# 检查并下载TinyStories数据
if [ ! -f "TinyStoriesV2-GPT4-train.txt" ]; then
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
fi
if [ ! -f "TinyStoriesV2-GPT4-valid.txt" ]; then
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
fi

# 检查并下载OpenWebText数据
if [ ! -f "owt_train.txt" ]; then
    wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
    gunzip owt_train.txt.gz
fi
if [ ! -f "owt_valid.txt" ]; then
    wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
    gunzip owt_valid.txt.gz
fi

cd ..
```

## 常见陷阱
1. **适配器函数**: 必须实现`tests/adapters.py`中的所有`NotImplementedError`函数
2. **状态字典键**: 必须与测试期望的完全匹配（参见适配器函数文档字符串）
3. **RoPE维度**: 注意力头维度使用`d_model // num_heads`
4. **数值稳定性**: RMSNorm计算使用float32，然后转换回原类型
5. **BPE训练**: 遵循GPT-2预分词模式（`tokenizer.py`中的`PAT`）

## 代码风格
- 行长度：120字符（ruff）
- 尽可能使用`einops`进行张量操作
- 新组件遵循`modules.py`中的现有模式
- 所有公共函数都需要类型注解