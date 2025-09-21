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


# Tests

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