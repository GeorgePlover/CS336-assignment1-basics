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