import regex as re
from typing import Tuple, Dict, List
from sortedcontainers import SortedSet


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    '''
        A BPE tokenizer
    '''
    def __init__(self):
        self.trained = False
        self.token_id_to_bytes = [bytes((i,)) for i in range(256)]
        self.merges = []
    
    def _read_string(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            string = f.read()
        return string
    
    def _split_by_special(self, string:str, special_tokens: List[str]) -> str:
        return re.split("|".join([re.escape(token) for token in special_tokens]), string)
    
    def _str_to_bytes_tuple(self, string: str) -> tuple:
        return tuple(string.encode("utf-8"))
    
    def _pre_tokenize_dict(self, string: str | List[str]) -> dict:
        pre_tokenized_dict = {}
        if isinstance(string, str):
            string = [string]
        strings = string
        for string in strings:
            for match in re.finditer(pattern=PAT, string=string):
                match_bytes = self._str_to_bytes_tuple(match.group())
                if match_bytes not in pre_tokenized_dict:
                    pre_tokenized_dict[match_bytes] = 1
                else:
                    pre_tokenized_dict[match_bytes] += 1

        return pre_tokenized_dict
    
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
            
            
    
    def _pre_tokenize(self, datafile: str, special_tokens: List[str]) -> Dict[Tuple, int]:
        '''
            return a pre_tokenized_dict like: {(104, 101, 108, 108, 111) : 3} => means 'hello' appear 3 times 
        '''
        for token in special_tokens:
            self.token_id_to_bytes.append(token.encode("utf-8"))
        
        string = self._read_string(filepath=datafile)
        string = self._split_by_special(string, special_tokens)
        pre_tokenized_dict = self._pre_tokenize_dict(string=string)
        
        return pre_tokenized_dict
         
    def train(self, datafile: str, special_tokens: List[str], vocab_end_size: int):
        pre_tokenized_dict = self._pre_tokenize(datafile, special_tokens)
        stepnum = vocab_end_size - len(self.token_id_to_bytes)
        # for i in range(stepnum):
        #     self._naive_train_step(pre_tokenized_dict=pre_tokenized_dict)
        
        self._optimized_train_steps(pre_tokenized_dict, stepnum)
                

    
if __name__ == "__main__":
    # from tests.test_train_bpe import test_train_bpe, test_train_bpe_speed, test_train_bpe_special_tokens
    # test_train_bpe_speed()
    
    # tokenizer = Tokenizer()
    # tokenizer.train(datafile="tests/fixtures/corpus.en",
    #                 special_tokens=["<|endoftext|>"],
    #                 vocab_end_size=260)
    pass