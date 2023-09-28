import moxing as mox
mox.file.copy_parallel('obs://liuyuanqiu/mindformers/data/chinese-corpus/comments2019.zip', '/home/ma-user/work/mindformers/data/chinesecorpus1/comments2019.zip')
mox.file.copy_parallel('obs://liuyuanqiu/mindformers/data/chinese-corpus/news2016zh_corpus.zip', '/home/ma-user/work/data/chinesecorpus1/news2016zh_corpus.zip')
mox.file.copy_parallel('obs://liuyuanqiu/mindformers/data/chinese-corpus/webText2019zh_corpus2.zip', '/home/ma-user/work/data/chinesecorpus1/webText2019zh_corpus2.zip')
mox.file.copy_parallel('obs://liuyuanqiu/mindformers/data/chinese-corpus/wiki_zh_2019.zip', '/home/ma-user/work/data/chinesecorpus1/wiki_zh_2019.zip')
mox.file.copy_parallel('obs://liuyuanqiu/mindformers/data/chinese-corpus/chinese-corpus-huagong.zip', '/home/ma-user/work/data/chinesecorpus1/chinese-corpus-huagong.zip')
mox.file.copy_parallel('obs://liuyuanqiu/mindformers/ckpts/llama2-tokenizer/tokenizer.model', '/home/ma-user/work/ckpts/llama2-tokenizer/tokenizer.model')