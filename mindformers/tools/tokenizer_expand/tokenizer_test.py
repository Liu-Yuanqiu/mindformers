from mindformers.models import LlamaTokenizer

LLAMA_TOKENIZER = '/home/ma-user/work/ckpts/llama2-tokenizer/'
CHINESE_LLAMA_TOKENIZER = '/home/ma-user/work/ckpts/chinese-llama2-tokenizer/'
llama2_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_TOKENIZER)
chinese_llama2_tokenizer = LlamaTokenizer.from_pretrained(CHINESE_LLAMA_TOKENIZER)

text = "[提要] 日前，美国《国家利益》杂志网站刊登了英国诺丁汉大学中国问题资深研究员迈科尔•科尔的《台湾如何在战争中打败大陆》的文章(下称《科文》)，为台独势力用战争手段与中国大陆对抗出谋划策。"
print(f"Tokenized by LLaMA tokenizer:\n{len(llama2_tokenizer.encode(text))}\n,{llama2_tokenizer.tokenize(text)}\n,{llama2_tokenizer.encode(text)}")
print(f"Tokenized by MedChat tokenizer:\n{len(chinese_llama2_tokenizer.encode(text))},{chinese_llama2_tokenizer.tokenize(text)}\n,{chinese_llama2_tokenizer.encode(text)}")
# print(f"chinese llama2 vocab size:{chinese_llama2_tokenizer.__dict__}")