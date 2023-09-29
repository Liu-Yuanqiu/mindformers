import os
import time
import glob
import json
import argparse
from tqdm import tqdm
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from mindformers.models import LlamaTokenizer

count = 0
# Step1 从四个中文数据集中取出长度100以上的文本
TXT_PATH = "/home/ma-user/work/data/chinesecorpus/corpus_zh.txt"
if not os.path.exists(TXT_PATH):
    WIKI_PATH = "/home/ma-user/work/data/chinesecorpus/wiki_zh/*/*"
    corpus = open(TXT_PATH, 'w', encoding='utf-8')
    for file in glob.glob(WIKI_PATH):
        print(f"正在处理文件{file}")
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                d = data["text"]
                if len(d)>100:
                    corpus.write(d + "\n")
                    count += 1
    INPUT_PATHS = ["/home/ma-user/work/data/chinesecorpus/comments2019/*.txt"]
    for path in INPUT_PATHS:
        print(f"正在处理文件夹{path}")
        for file in glob.glob(path):
            print(f"正在处理文件{file}")
            with open(file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    d = line.strip()
                    if len(d)>100:
                        corpus.write(d + "\n")
                        count += 1
    corpus.close()
else:
    count = len(open(TXT_PATH, 'r', encoding='utf-8').readlines())
print(f"数据集大小为{count}")

# Step2 训练tokenizer
CHINESE_TOKENIZER = '/home/ma-user/work/ckpts/chinese-tokenizer/'
if not os.path.exists(CHINESE_TOKENIZER):
    os.mkdir(CHINESE_TOKENIZER)
print("开始训练...")
start_time = time.time()
spm.SentencePieceTrainer.train(
    input=TXT_PATH,
    model_prefix=CHINESE_TOKENIZER + "tokenizer",  # 模型前缀
    shuffle_input_sentence=True,  # 是否打乱句子
    train_extremely_large_corpus=True,
    max_sentence_length=8192,  # 句子最大长度,4096*2,一个中文字符长度2
    model_type="BPE",
    vocab_size=32000,
    character_coverage=1,
    split_digits=True,
    split_by_unicode_script=True,
    byte_fallback=True,
    allow_whitespace_only_pieces=True,
    remove_extra_whitespaces=False,
    normalization_rule_name="nfkc",
    num_threads=96,
    seed_sentencepiece_size=count,
    input_sentence_size=count,
)
print(f"训练结束，用时{(time.time()-start_time)/60}分钟。")

# Step2.1 测试中文分词效果
sp = spm.SentencePieceProcessor()
sp.load(CHINESE_TOKENIZER + '.model')
text = "甲苯歧化和烷基转移技术起始于美国UOP公司于60年代末开发的Tatoray工艺，不过使用的TA系列催化剂存在寿命较短的问题，后续开发的TAC工艺[1]是一种更为成熟的工艺，采用固定床反应器和丝光沸石催化剂，可以将C和C10重质芳烃有效转化为二甲苯，且流程较为简单。"
print(sp.encode_as_pieces(text))
print(sp.encode_as_ids(text))

# Step3 合并中英文tokenizer
LLAMA_TOKENIZER = '/home/ma-user/work/ckpts/llama2-tokenizer/'
llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_TOKENIZER)
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(CHINESE_TOKENIZER + '.model')

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.s.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
print(llama_tokenizer.__dict__)
print(f"llama2词表长度：{llama_tokenizer.vocab_size}")
print(f"中文词表长度：{len(chinese_sp_model)}")

llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before llama length:{len(llama_spm_tokens_set)}")
if "<pad>" not in llama_spm_tokens_set:
    new_p = sp_pb2_model.ModelProto().SentencePiece()
    new_p.piece = "<pad>"
    new_p.score = 0
    llama_spm.pieces.append(new_p)
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New llama model pieces: {len(llama_spm.pieces)}")

## Step4 保存合并后分词器
CHINESE_LLAMA_TOKENIZER = '/home/ma-user/work/ckpts/chinese-llama2-tokenizer/'
if not os.path.exists(CHINESE_LLAMA_TOKENIZER):
    os.mkdir(CHINESE_LLAMA_TOKENIZER)
with open(CHINESE_LLAMA_TOKENIZER + 'tokenizer.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
print(f"Merged tokenizer has been saved to {CHINESE_LLAMA_TOKENIZER}")

# Step4.1 测试分词效果
llama2_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_TOKENIZER)
chinese_llama2_tokenizer = LlamaTokenizer.from_pretrained(CHINESE_LLAMA_TOKENIZER)
print(f"Tokenized by LLaMA tokenizer:\n{len(llama_tokenizer.encode(text))},{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by MedChat tokenizer:\n{len(chinese_llama2_tokenizer.encode(text))},{chinese_llama2_tokenizer.tokenize(text)}")
