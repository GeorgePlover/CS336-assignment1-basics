import regex as re
from typing import Tuple, Dict, List, Any
from sortedcontainers import SortedSet
from functools import lru_cache

from cs336_basics.profiler import profile
from cs336_basics.pretokenization_example import find_chunk_boundaries

from multiprocessing import Pool
import numpy as np
import numpy.typing as npt
import sys
import io


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_dict_sum(dict1: Dict, dict2: Dict) -> Dict:
    # 启发式合并
    if len(dict2) > len(dict1):
        dict1, dict2 = dict2, dict1
    for key in list(dict2.keys()):
        val = dict2.pop(key)
        if key not in dict1:
            dict1[key] = val
        else:
            dict1[key] += val
    return dict1        

class Tokenizer:
    '''
        A BPE tokenizer
    '''
    def __init__(self):
        self.trained = False
        self.token_id_to_bytes = [bytes((i,)) for i in range(256)]
        self.merges = []
        self.special_tokens = []
        self.vocab = {}
        self.indexed_merges = None # (int,int) -> int (index in merges)
        self.merged_token_id = None # (int,int) -> int (merged new vocab token id)
        self.inv_vocab = None # bytes -> int
        
    def init_from_given_member(self, 
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
        ):
        self.trained = True
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.merges = merges
        
    
    def _read_string(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            string = f.read()
        return string
    
    def _stream_split_by_special(self, string:str, special_tokens: List[str]):
        pattern = re.compile("|".join([re.escape(token) for token in special_tokens]))
        last_end = 0
        for match in pattern.finditer(string):
            start = match.start()
            yield string[last_end:start]  # 分割出的片段
            last_end = match.end()
        yield string[last_end:]  # 最后一段
    
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
    
    def _str_to_bytes_tuple(self, string: str) -> tuple:
        if self.inv_vocab is not None: # 使用给定的编码方案
            return  tuple([ self.inv_vocab[bytes([b])] for b in string.encode("utf-8")])
        return tuple(string.encode("utf-8"))
           
    def _pre_tokenize_dict(self, string: str, special_tokens: List[str] = ["<|endoftext|>"]) -> dict:

        pre_tokenized_dict = {}
        
        for string in self._stream_split_by_special(string, special_tokens):
            for match in re.finditer(pattern=PAT, string=string):
                match_bytes = self._str_to_bytes_tuple(match.group())
                if match_bytes not in pre_tokenized_dict:
                    pre_tokenized_dict[match_bytes] = 1
                else:
                    pre_tokenized_dict[match_bytes] += 1

        print(f"A pre_tokenized_dict with length {len(pre_tokenized_dict)} have been done.")
        return pre_tokenized_dict
    
    def _pre_tokenize_multi_processes(self, filename: str, num_processes: int,
                                           split_special_token: str = "<|endoftext|>", # 用来切分多进程分割
                                           special_tokens: List[str] = ["<|endoftext|>"] # 用来处理所有特殊token
                                    ):
        res_dict = {}
        with open(filename, "rb") as f:
            print (f"Start to find {num_processes} boundaries...")
            boundaries = find_chunk_boundaries(f, num_processes, bytes(split_special_token.encode("utf-8")))
            print (f"{len(boundaries)} boundaries found.\nStart multi processes.")

                # The following is a serial implementation, but you can parallelize this
                # by sending each start/end pair to a set of processes.
            
            with Pool(processes=len(boundaries)) as pool:  
                args = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    print("chunk size:", sys.getsizeof(chunk) / (2**30), "GiB")
                    
                    args.append((chunk, special_tokens))
                    
                results = pool.starmap(self._pre_tokenize_dict, args)
                merge_cnt = 0
                
                for _dict in results:
                    res_dict = merge_dict_sum(res_dict, _dict)
                    merge_cnt += 1
                    print(f"Merged {merge_cnt} process(es).")
        
        return res_dict
                    
    def _token_pair_to_cmp(self, pair):
        return (self.token_id_to_bytes[pair[0]], self.token_id_to_bytes[pair[1]])
    
    def _naive_train_step(self, pre_tokenized_dict: Dict[Tuple, int]) -> None:
        '''
            Count the times of each pair appearance and merge the most one. 
        '''
        pairs_cnt = {}
        best_pair = None
        pre_tokenized_list = list(pre_tokenized_dict.items())
        print(pre_tokenized_list)
        for word in pre_tokenized_list:
            word_key, word_val = word
            for i in range(len(word_key)-1):
                pair = (word_key[i], word_key[i+1])
                if pair not in pairs_cnt:
                    pairs_cnt[pair] = word_val
                else:
                    pairs_cnt[pair] += word_val
                if (best_pair == None 
                    or pairs_cnt[best_pair] < pairs_cnt[pair]
                    or (pairs_cnt[best_pair] == pairs_cnt[pair] 
                        and self._token_pair_to_cmp(best_pair) < self._token_pair_to_cmp(pair)
                        )
                    ):
                    best_pair = pair

        # debug = [(pairs_cnt[key],bytes([i%256 for i in key])) for key in pairs_cnt.keys()]
        # debug.sort()
        # print(debug[-5:])
        
        if best_pair == None:
            print(f"My_Warning: When vocab size is {len(self.token_id_to_bytes)}, cannot find more pair.")
            return
        
        # record merges
        self.merges.append((self.token_id_to_bytes[best_pair[0]], self.token_id_to_bytes[best_pair[1]]))
        
        # add new token
        new_bytes = self.token_id_to_bytes[best_pair[0]] + self.token_id_to_bytes[best_pair[1]]
        new_token_id = len(self.token_id_to_bytes)
        self.token_id_to_bytes.append(new_bytes)
        
        # do merge
        for word in pre_tokenized_list:
            word_key, word_val = word
            flag = False
            for i in range(len(word_key)-1):
                pair = (word_key[i], word_key[i+1])
                if pair == best_pair:
                    flag = True
                    break
            if (flag):
                new_word = []
                i = 0
                while (i < len(word_key)):
                    if (i != len(word_key) - 1) and (best_pair == (word_key[i], word_key[i+1])):
                        new_word.append(new_token_id)
                        i += 2
                    else:
                        new_word.append(word_key[i])
                        i += 1
                new_word = tuple(new_word)
                pre_tokenized_dict[new_word] = pre_tokenized_dict.pop(word_key)
    
    def _count_a_word(self, word: Tuple, word_cnt: int,
                      pair_cnt: Dict[Tuple[int, int], int],
                      pair_belong_word_indexs: Dict[Tuple[int, int], list],
                      word_index: int):
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            if pair not in pair_cnt:
                pair_cnt[pair] = word_cnt
            else:
                pair_cnt[pair] += word_cnt
                
            if pair not in pair_belong_word_indexs:
                pair_belong_word_indexs[pair] = [word_index]
            else:
                pair_belong_word_indexs[pair].append(word_index)
    
    def _optimized_train_steps(self, pre_tokenized_dict: Dict[Tuple, int], train_step_num: int):
        '''
            index the count to find the best pair
        '''
        # init
        pair_cnt = {}
        num_indexed_pair_cnt = SortedSet()
        pair_belong_word_indexs = {}
        
        pre_tokenized_list = list(pre_tokenized_dict.items())
        
        for idx in range(len(pre_tokenized_list)):
            word_key, word_val = pre_tokenized_list[idx]
            self._count_a_word(word=word_key, word_cnt=word_val, pair_cnt=pair_cnt,
                               pair_belong_word_indexs=pair_belong_word_indexs,
                               word_index=idx)
        
        for pair, cnt in pair_cnt.items():
            num_indexed_pair_cnt.add((cnt, self._token_pair_to_cmp(pair), pair))
        
        # start train loop
        for i in range(train_step_num):
            max_cnt, _not_used, best_pair = num_indexed_pair_cnt.pop()
            
            # record merges
            self.merges.append((self.token_id_to_bytes[best_pair[0]], self.token_id_to_bytes[best_pair[1]]))
            
            # add new token
            new_bytes = self.token_id_to_bytes[best_pair[0]] + self.token_id_to_bytes[best_pair[1]]
            new_token_id = len(self.token_id_to_bytes)
            self.token_id_to_bytes.append(new_bytes)
            
            # maintain the data structure
            for wid in pair_belong_word_indexs[best_pair]:
                word, num = pre_tokenized_list[wid]
                
                new_word = []
                i = 0
                flag = False
                while (i < len(word)):
                    if (i != len(word) - 1) and (best_pair == (word[i], word[i+1])):
                        new_word.append(new_token_id)
                        i += 2
                        flag = True # really happens to merge
                    else:
                        new_word.append(word[i])
                        i += 1
                
                if flag == False:
                    continue        
                
                new_word = tuple(new_word)
                
                previous_pair_cnt = {}
                previous_pair_belong = {}
                self._count_a_word(word, num, previous_pair_cnt, previous_pair_belong, wid)
                
                new_pair_cnt = {}
                new_pair_belong = {}
                self._count_a_word(new_word, num, new_pair_cnt, new_pair_belong, wid)
                
                # maintain the cnt of removal pairs (exclude the best pair)
                for key in previous_pair_cnt:
                    if key == best_pair:
                        continue
                    
                    decrease = 0
                    if key not in new_pair_cnt:
                        decrease = previous_pair_cnt[key]
                    elif previous_pair_cnt[key] != new_pair_cnt[key]: 
                        decrease =  previous_pair_cnt[key] - new_pair_cnt[key]
                    
                    if decrease != 0:
                        num_indexed_pair_cnt.remove((pair_cnt[key], self._token_pair_to_cmp(key),  key))
                        pair_cnt[key] -= decrease
                        num_indexed_pair_cnt.add((pair_cnt[key], self._token_pair_to_cmp(key),  key))
                
                # maintain the cnt of newly pairs
                for key in new_pair_cnt:
                    
                    increase = 0
                    if key not in previous_pair_cnt:
                        increase = new_pair_cnt[key]
                    
                    if increase != 0:
                        if key not in pair_cnt:
                            pair_cnt[key] = 0
                            num_indexed_pair_cnt.add((0, self._token_pair_to_cmp(key), key))
                            pair_belong_word_indexs[key] = []
                        # TODO: the same index may append multi-times, can be optimized?
                        pair_belong_word_indexs[key].append(wid)
                        num_indexed_pair_cnt.remove((pair_cnt[key], self._token_pair_to_cmp(key), key))
                        pair_cnt[key] += increase
                        num_indexed_pair_cnt.add((pair_cnt[key], self._token_pair_to_cmp(key), key))
                        
                # update the new word
                pre_tokenized_list[wid] = (new_word, pre_tokenized_list[wid][1])
                
            pair_cnt.pop(best_pair)
            pair_belong_word_indexs.pop(best_pair)
        
        # update the pre_tokenized_dict
        pre_tokenized_dict = dict(pre_tokenized_list)
            
            
    
    def _pre_tokenize(self, datafile: str, special_tokens: List[str],
                      multi_processes:bool = False, num_processes:int = 1) -> Dict[Tuple, int]:
        '''
            return a pre_tokenized_dict like: {(104, 101, 108, 108, 111) : 3} => means 'hello' appear 3 times 
        '''
        for token in special_tokens:
            self.token_id_to_bytes.append(token.encode("utf-8"))

        if multi_processes:
            pre_tokenized_dict = self._pre_tokenize_multi_processes(datafile, num_processes,
                                               "<|endoftext|>",
                                               special_tokens)
        else:
            string = self._read_string(filepath=datafile)
            pre_tokenized_dict = self._pre_tokenize_dict(string=string, special_tokens=special_tokens)
        
        return pre_tokenized_dict
    
    def test_multi_process_pre_tokenize(self):
        special_tokens = ["<|endoftext|>"]
        file_path = "data/TinyStoriesV2-GPT4-valid.txt"
        
        import time
        start = time.time()
        res_single_process = self._pre_tokenize(file_path, special_tokens)
        print("single process time cost:", time.time() - start)
        start = time.time()
        res_multi_processes = self._pre_tokenize_multi_processes(file_path, 8, "<|endoftext|>", special_tokens)
        print("multi processes time cost:", time.time() - start)
        
        assert set(res_single_process) == set(res_multi_processes)
        print("test_multi_process_pre_tokenize passed")
         
    def train(self, datafile: str, special_tokens: List[str], vocab_end_size: int,
              multi_processes:bool = True, num_processes:int = 8):
        pre_tokenized_dict = self._pre_tokenize(datafile, special_tokens, multi_processes, num_processes)
        stepnum = vocab_end_size - len(self.token_id_to_bytes)
        # for i in range(stepnum):
        #     self._naive_train_step(pre_tokenized_dict=pre_tokenized_dict)
        
        self._optimized_train_steps(pre_tokenized_dict, stepnum)
        self.trained = True
        self.vocab = dict(enumerate(self.token_id_to_bytes))
        self.special_tokens = special_tokens

    @lru_cache(maxsize=1024)
    def _merge_single_word(self, token_tuple: Tuple[int]):
        
        assert(len(token_tuple)>0)
        
        if len(token_tuple) == 1:
            return token_tuple
        
        while True:
            pair_list = []
            for a,b in zip(token_tuple, token_tuple[1:]):
                if (a,b) in self.indexed_merges:
                    pair_list.append((self.indexed_merges[(a,b)], (a,b)))
            if len(pair_list) == 0:
                break
            
            aim_merge_id, aim_merge = min(pair_list)
            
            new_tuple = []
            i = 0
            while i < len(token_tuple):
                if (i+1 < len(token_tuple)) and ((token_tuple[i], token_tuple[i+1]) == aim_merge):
                    new_tuple.append(self.merged_token_id[(token_tuple[i], token_tuple[i+1])])
                    i += 2
                else:
                    new_tuple.append(token_tuple[i])
                    i += 1
            token_tuple = new_tuple
        
        return token_tuple
    
    def _init_indexed_merges(self):
        inv_vocab = dict([(value, key) for key, value in list(self.vocab.items())])
        self.indexed_merges = dict([
            ((inv_vocab[value[0]], inv_vocab[value[1]]), key) 
            for key, value in enumerate(self.merges)
        ])
        self.merged_token_id = dict([
            ((inv_vocab[a], inv_vocab[b]), inv_vocab[a+b]) 
            for a,b in self.merges
        ])
        self.inv_vocab = inv_vocab
        
    def encode_iterable(self, string: str):
        if (self.indexed_merges is None) or (self.merged_token_id is None):
            self._init_indexed_merges()
        
        if self.special_tokens is None:
            self.special_tokens = []
        
        for word in self._stream_split_with_special(string, self.special_tokens):
            if word not in self.special_tokens: # aim
                bytes_word = self._str_to_bytes_tuple(word)
                merged_word = self._merge_single_word(bytes_word)
                for ret in merged_word:
                    yield ret
            else: # a special token
                ret = self.inv_vocab[word.encode("utf-8")]
                yield ret
        
    def encode(self, string: str)-> list:
        return [res for res in self.encode_iterable(string)]
    
    def decode(self, ids: list)-> str:
        return b''.join([self.vocab[token] for token in ids]).decode(encoding="utf-8",errors="ignore")
        
       
@profile
def main():
    from tests.adapters import run_train_bpe
    vocab, merges = run_train_bpe(input_path="data/owt_train.txt",
                                  vocab_size=32000,
                                  special_tokens=["<|endoftext|>"]
                                  )
    with open("trained_vocab.txt","w") as f:
        f.write(str(vocab))
        f.close()
    with open("trained_merges.txt","w") as f:
        f.write(str(merges))
        f.close()         

    
if __name__ == "__main__":
    # from tests.test_train_bpe import test_train_bpe, test_train_bpe_speed, test_train_bpe_special_tokens
    # test_train_bpe_speed()
    
    # tokenizer = Tokenizer()
    # tokenizer.test_multi_process_pre_tokenize()
    
    from tests.test_tokenizer import test_encode_memory_usage,test_encode_iterable_tinystories_sample_roundtrip,test_roundtrip_single_character
    test_encode_memory_usage()
    # test_roundtrip_single_character()
    # test_encode_iterable_tinystories_sample_roundtrip()
    
    # main()
    pass