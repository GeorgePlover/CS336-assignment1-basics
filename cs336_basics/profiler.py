import cProfile
import pstats
import functools

def profile(func):
    @functools.wraps(func)  # 保持原函数元信息
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        # 执行原函数
        result = func(*args, **kwargs)
        
        pr.disable()
        
        # 打印性能分析结果
        ps = pstats.Stats(pr)
        ps.sort_stats('cumtime')  # 按累计时间排序
        ps.print_stats(10)  # 只显示前10行
        
        return result
    return wrapper

# 使用示例
# @profile
# def slow_function(n):
#     return sum(i ** 2 for i in range(n))

# slow_function(1000000)