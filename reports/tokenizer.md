# Train on owt_train

```bash
20290201618 function calls (20290182003 primitive calls) in 9755.046 seconds

   Ordered by: cumulative time
   List reduced from 4394 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.379    0.379 9741.307 9741.307 /nas/user/nsh/projects/CS336-assignment1-basics/tests/adapters.py:566(run_train_bpe)
        1    6.277    6.277 9740.927 9740.927 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:319(train)
        1 4500.856 4500.856 8788.389 8788.389 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:180(_optimized_train_steps)
     2304    0.519    0.000 1844.616    0.801 /home/nsh/miniconda3/lib/python3.12/multiprocessing/pool.py:333(_maintain_pool)
12504237253/12504236934 1620.444    0.000 1620.444    0.000 {built-in method builtins.len}
     4611    0.035    0.000 1585.209    0.344 /home/nsh/miniconda3/lib/python3.12/multiprocessing/connection.py:1121(wait)
     4611    0.031    0.000 1572.595    0.341 /home/nsh/miniconda3/lib/python3.12/selectors.py:402(select)
6073853287 1059.705    0.000 1059.705    0.000 {method 'append' of 'list' objects}
       17    0.494    0.029 1002.464   58.968 /home/nsh/miniconda3/lib/python3.12/multiprocessing/connection.py:202(send)
     2304    0.012    0.000  969.879    0.421 /home/nsh/miniconda3/lib/python3.12/multiprocessing/pool.py:289(_join_exited_workers)


         20290201618 function calls (20290182003 primitive calls) in 9755.046 seconds

   Ordered by: internal time
   List reduced from 4394 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1 4500.856 4500.856 8788.389 8788.389 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:180(_optimized_train_steps)
12504237253/12504236934 1620.444    0.000 1620.444    0.000 {built-in method builtins.len}
6073853287 1059.705    0.000 1059.705    0.000 {method 'append' of 'list' objects}
     4611  743.650    0.161  803.983    0.174 {method 'poll' of 'select.poll' objects}
 78664796  290.705    0.000  312.711    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:164(_count_a_word)
227747508  202.025    0.000  202.025    0.000 {built-in method _bisect.bisect_left}
113873754  184.296    0.000  707.386    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/.venv/lib/python3.12/site-packages/sortedcontainers/sortedset.py:456(remove)
117505633  178.851    0.000  178.851    0.000 {built-in method _bisect.insort_right}
117505926  143.387    0.000  469.804    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/.venv/lib/python3.12/site-packages/sortedcontainers/sortedlist.py:253(add)
113873754  141.623    0.000  476.072    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/.venv/lib/python3.12/site-packages/sortedcontainers/sortedlist.py:426(remove)
```

# Train on TinyStories-train

* 没写并行优化

* profile:

```bash
(.venv) (base) nsh@node28:/nas/user/nsh/projects/CS336-assignment1-basics$  cd /nas/user/nsh/projects/CS336-assignment1-basics ; /usr/bin/env /nas/user/nsh/projects/CS336-assignment1-basics/.venv/bin/python /home/nsh/.vscode-server/extensions/ms-python.debugpy-2025.13.2025091201-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 37121 -- cs336_basics/tokenizer.py 
         1686562114 function calls (1683714219 primitive calls) in 1314.196 seconds

   Ordered by: cumulative time
   List reduced from 4231 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
19551/13029    1.231    0.000 2045.910    0.157 /home/nsh/miniconda3/lib/python3.12/queue.py:154(get)
26071/26069    3.849    0.000 1305.707    0.050 /home/nsh/miniconda3/lib/python3.12/threading.py:323(wait)
13042/13041    0.721    0.000  559.384    0.043 /home/nsh/miniconda3/lib/python3.12/threading.py:637(wait)
536592168  258.280    0.000  378.238    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:26(_str_to_bytes_tuple)
536592272  119.958    0.000  119.958    0.000 {method 'encode' of 'str' objects}
536592168  105.480    0.000  105.480    0.000 {method 'group' of '_regex.Match' objects}
  2717700    3.826    0.000   43.728    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/.venv/lib/python3.12/site-packages/regex/regex.py:340(finditer)
  2717701   14.283    0.000   37.141    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/.venv/lib/python3.12/site-packages/regex/regex.py:449(_compile)
       12    0.001    0.000   16.104    1.342 /nas/user/nsh/projects/CS336-assignment1-basics/.venv/lib/python3.12/site-packages/torch/_jit_internal.py:1025(_overload)
  5435635    7.772    0.000   15.513    0.000 /home/nsh/miniconda3/lib/python3.12/enum.py:1541(__and__)
```

* 使用python多进程优化，16进程，加速显著，大约总耗时2min：

```bash
  27399047 function calls (27366788 primitive calls) in 131.048 seconds

   Ordered by: cumulative time
   List reduced from 4340 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       33    0.135    0.004  201.116    6.094 /home/nsh/miniconda3/lib/python3.12/multiprocessing/connection.py:202(send)
       43    0.011    0.000  183.362    4.264 /home/nsh/miniconda3/lib/python3.12/multiprocessing/pool.py:500(_wait_for_updates)
 2411/656    3.023    0.001  137.923    0.210 /home/nsh/miniconda3/lib/python3.12/threading.py:323(wait)
 1219/331    0.499    0.000  117.459    0.355 /home/nsh/miniconda3/lib/python3.12/threading.py:637(wait)
       38    0.000    0.000   99.463    2.617 /home/nsh/miniconda3/lib/python3.12/multiprocessing/connection.py:406(_send_bytes)
       54    0.000    0.000   99.462    1.842 /home/nsh/miniconda3/lib/python3.12/multiprocessing/connection.py:381(_send)
        1    0.457    0.457   99.421   99.421 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:278(_pre_tokenize)
        1    0.000    0.000   98.962   98.962 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:61(_pre_tokenize_multi_processes)
        1    0.001    0.001   98.961   98.961 /home/nsh/miniconda3/lib/python3.12/multiprocessing/pool.py:738(__exit__)
       18    0.000    0.000   98.876    5.493 /home/nsh/miniconda3/lib/python3.12/multiprocessing/util.py:205(__call__)


         27399047 function calls (27366788 primitive calls) in 131.048 seconds

   Ordered by: internal time
   List reduced from 4340 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   9703/3   86.677    0.009    0.000    0.000 {method 'acquire' of '_thread.lock' objects}
       38    7.301    0.192    7.301    0.192 {method 'dump' of '_pickle.Pickler' objects}
     5013    6.150    0.001    6.150    0.001 {built-in method posix.stat}
      250    4.249    0.017    4.249    0.017 {method 'decode' of 'bytes' objects}
 2411/656    3.023    0.001  137.923    0.210 /home/nsh/miniconda3/lib/python3.12/threading.py:323(wait)
      813    1.893    0.002    1.893    0.002 {built-in method _io.open_code}
10249243/10248965    1.373    0.000    1.373    0.000 {built-in method builtins.len}
   615493    1.371    0.000    1.521    0.000 /nas/user/nsh/projects/CS336-assignment1-basics/cs336_basics/tokenizer.py:156(_count_a_word)
    12492    1.273    0.000    1.273    0.000 {built-in method posix.lstat}
      844    1.195    0.001    1.195    0.001 {method 'read' of '_io.BufferedReader' objects}
```

# Train Tests

## test_train_bpe(Passed)

* 实现了基于索引优化的merge。利用一个SortedSet维护最优的pair，并且利用字典存放每个pair所归属的单词，这样可以彻底避免遍历所有单词。

* 在test_train_bpe_speed测试上大约用时 0.40 sec

```bash                                                                                       
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

===============================================
 3 passed in 3.21s
================================================
```

# Encode

## 流式处理输入

核心代码如下。

* _buffer_read：首先是读取，一次处理一个read_size里面的字符串，这里把文件读入和直接的字符串读入都归一处理了，增强后续代码的复用。

* _stream_split_with_special：关键点。设置一个保留的buffer_size，他必须大于最长的special_token和单个单词的长度，用来防止token或单词被块状read给截断。调用正则匹配的时候，套用两层，一层处理special_tokens，一层处理words，都要注意保留足够的buffer长度来预防截断。

```python
     def _buffer_read(self, stream: str | io.FileIO, read_size:int = 8192)-> Any:
        if isinstance(stream, str):
            return stream[:read_size], stream[read_size:]
        else:
            string = stream.read(read_size)
            return string, stream
        
    def _stream_split_with_special(self, stream:str, special_tokens: List[str] | None, buffer_size:int = 1024):
        '''
            change stream into word/special_token
        '''
        if special_tokens != None:
            special_tokens.sort(key=len, reverse=True)
        else:
            special_tokens = []
        
        pattern_special = re.compile("|".join([re.escape(token) for token in special_tokens]))
        pattern_word = re.compile(PAT)
        
        buffer = ""
        
        while True:
            string, stream = self._buffer_read(stream)
            buffer += string
            if string == "":
                buffer_size = 0 # 直接处理完最后一轮，不留buffer
            
            last = 0
            if len(special_tokens) != 0:
                for match in pattern_special.finditer(buffer):
                    for word in pattern_word.findall(buffer[last: match.start()]):
                        yield word
                        
                    if len(buffer) - match.start() < buffer_size: # 防止special_token来自于被截断的更长的special_token
                        last = match.start()
                        break
                    
                    yield match.group()
                    last = match.end()
                
            buffer = buffer[last:]
            
            last = 0
            for match in pattern_word.finditer(buffer):
                
                if len(buffer) - match.end() < buffer_size: # 防止token被截断
                    last = match.start()
                    break
                
                yield match.group()
                
            buffer = buffer[last:]
            
            if buffer_size == 0:
                break
```