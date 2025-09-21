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