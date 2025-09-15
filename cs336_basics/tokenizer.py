import regex as re
from typing import Tuple, Dict, List

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    '''
        A BPE tokenizer
    '''
    def __init__(self):
        self.trained = False
        self.token_id_to_bytes = [chr(i) for i in range(256)]
    
    def _read_string(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            string = f.read()
        return string
    
    def _str_to_bytes_tuple(self, string: str) -> tuple:
        return tuple(string.encode("utf-8"))
    
    def _pre_tokenize_dict(self, string: str) -> dict:
        pre_tokenized_dict = {}
        for match in re.finditer(pattern=PAT, string=string):
            match_bytes = self._str_to_bytes_tuple(match.group())
            if match_bytes not in pre_tokenized_dict:
                pre_tokenized_dict[match_bytes] = 1
            else:
                pre_tokenized_dict[match_bytes] += 1
                
        return pre_tokenized_dict
    
    def _naive_train_step(self, pre_tokenized_dict: Dict[Tuple, int]) -> None:
        '''
            Count the times of each pair appearance and merge the most one. 
        '''
        pairs_cnt = {}
        best_pair = None
        pre_tokenized_list = list(pre_tokenized_dict.items())
        for word in pre_tokenized_list:
            word_key, word_val = word
            for i in range(len(word_key)-1):
                pair = (word_key[i], word_key[i+1])
                if pair not in pairs_cnt:
                    pairs_cnt[pair] = word_val
                else:
                    pairs_cnt[pair] += word_val
                if (best_pair == None or pairs_cnt[best_pair] < pairs_cnt[pair]
                        or (pairs_cnt[best_pair] == pairs_cnt[pair] and best_pair < pair)
                    ):
                    best_pair = pair
        
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
                        new_word.append(i)
                        i += 1
                new_word = tuple(new_word)
                pre_tokenized_dict[new_word] = pre_tokenized_dict.pop(word_key)
    
    def _pre_tokenize(self, datafile: str, special_tokens: List[str]) -> Dict[Tuple, int]:
        '''
            return a pre_tokenized_dict like: {(104, 101, 108, 108, 111) : 3} => means 'hello' appear 3 times 
        '''
        string = self._read_string(filepath=datafile)
        pre_tokenized_dict = self._pre_tokenize_dict(string=string)
        
        return pre_tokenized_dict
         
    def train(self, datafile: str, special_tokens: List[str], stepnum: int):
        pre_tokenized_dict = self._pre_tokenize(datafile, special_tokens)
        
        for i in range(stepnum):
            self._naive_train_step(pre_tokenized_dict=pre_tokenized_dict)
            print(self.token_id_to_bytes)
                

    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.train(datafile="tests/fixtures/tinystories_sample.txt", stepnum=100)
    
            
