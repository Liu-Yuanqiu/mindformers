# 欢迎来到MindSpore Transformers（MindFormers）
# Llama
## 0. 准备工作
### 0.1 notebook
- 8卡notebook
- 镜像mindformers：mindformers_0.6rc1_mindspore_2_0_modelarts

## 1. 代码结构介绍
  ```bash
  work
      ├── ckpts
            ├── chinese-llama2-tokenizer #添加中文单词后的分词模型
            ├── chinese-tokenizer        #从中文语料库得到的中文分词模型
            ├── llama2-tokenizer         #原始llama2英文分词模型
            ├── llama2-7b-pretrain       #使用添加中文单词的分词模型预训练
            ├── llama2-7b-lora           #预训练基础上进行lora微调
            └── llama2-7b                #原始llama2英文权重
      ├── data
            ├── chinesecorpus            #扩充词表所用中文语料
            ├── medchat                  #眩晕症所用数据
            └── knowledgegraph           #知识图谱所用数据
      └── mindformers
            ├── configs
            ├── mindformers
            ├── output
            ├── scripts
            ├── llama_infer.py
            ├── llama_test.py
            └── run_mindformers.py
  ```
## 2. 数据处理
### 2.0 llama2原始英文权重和分词器
```bash
# 本地执行
# 2.0.1 copy /home/ma-user/work/mindformers/mindformers/models/llama/convert_weight.py 到本地
# 2.0.2 下载llama2-7b原始权重
# 2.0.3 安装mindspore 2.0 cpu版本，将pytorch权重转换为mindspore权重文件，并上传到obs
python convert_weight.py --torch_ckpt_dir ./llama2-7b --mindspore_ckpt_path ./llama2-7b/llama2-7b.ckpt

# ModelArts notebook中执行
cd work
mkdir ckpts
cd ckpts
mkdir llama2-tokenizer
mkdir llama2-7b
python /home/ma-user/work/mindformers/mindformers/tools/tokenizer_expand/copy_llama2.py
```
### 2.1 中文预料数据拷贝，用于扩充llama2词表
```bash
cd work
mkdir data
cd data
mkdir chinesecorpus
python /home/ma-user/work/mindformers/mindformers/tools/tokenizer_expand/copy_chinese_corpus.py
unzip chinese-corpus-huagong.zip
unzip -d ./comments2019 comments2019.zip
unzip -d ./news2016 news2016zh_corpus.zip
unzip -d ./webtext2019 webText2019zh_corpus2.zip
unzip wiki_zh_2019.zip
```
### 2.2 Lora微调数据拷贝（待完善）
2.2.1 使用fasechat工具添加prompts模板，转换为多轮对话模式
```python
python /home/ma-user/work/mindformers/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py --data_path /home/ma-user/work/data/medchat/2500_data_v2.json --output_path /home/ma-user/work/data/medchat/2500_data_v2_conversation.json
```
2.2.2 将数据转换为mindrecord格式
```python
# 使用llama进行微调时，句子长度seq_length为2048
# 使用llama2进行微调时，句子长度seq_length需要设置为4096
python /home/ma-user/work/mindformers/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py --input_glob /home/ma-user/work/data/medchat/2500_data_v2_conversation.json --dataset_type qa --model_file /home/ma-user/work/ckpts/llama2-7b-pretrain/tokenizer.model --seq_length 4096 --output_file  /home/ma-user/work/data/medchat/xuanyun4096.train.mindrecord
```

## 3. llama2扩充词表预训练
```python
cd work/mindformers
# 3.1 扩充词表（需要大概8个小时）
python mindformers/tools/tokenizer_expand/tokenizer_expand.py
# 3.2 测试扩充后对中文编码能力（扩充后词表大小61045）
python mindformers/tools/tokenizer_expand/tokenizer_test.py
# 3.3 将中文预料转化为mindrecord格式
python mindformers/tools/dataset_preprocess/llama/llama_preprocess.py --dataset_type chinesecorpus --input_glob /home/ma-user/work/data/chinesecorpus/corpus_zh_sen0_len35294012.txt --model_file /home/ma-user/work/ckpts/chinese-llama2-tokenizer/tokenizer.model --seq_length 4096 --output_file /home/ma-user/work/data/chinesecorpus/corpus_zh.mindrecord
# 3.3 使用中文语料库预训练（训练wordembedding参数）
bash run_distribute.sh /user/config/nbstart_hccl.json /home/ma-user/work/mindformers/configs/llama_ailab/pretrain_llama2_7b.yaml [0,8] train
# 3.4 pipeline>1 进行权重合并，转移权重文件
python mindformers/tools/move_ckpt.py --ckpt_pre_name="llama2_7b_pretrain_rank_" --ckpt_post_name="-500_1"
# 3.5 合并权重
python mindformers/tools/transform_ckpt.py --src_ckpt_strategy /home/ma-user/work/mindformers/output/strategy/ --src_ckpt_dir /home/ma-user/work/mindformers/output/ckpt/ --dst_ckpt_dir /home/ma-user/work/ckpts/llama2-7b-pretrain/ --prefix llama2-7b-pretrain-1001
# 3.6 测试
python llama_test.py --model="/home/ma-user/work/mindformers/configs/llama_ailab/predict_llama2_7b_pretrain.yaml" --checkpoint_path="/home/ma-user/work/ckpts/llama2-7b-pretrain/rank_0/llama2-7b-pretrain-1001.ckpt"
```

## 4. llama2 Lora微调
```python
cd work/mindformers/scripts
bash run_distribute.sh /user/config/nbstart_hccl.json /home/ma-user/work/mindformers/configs/llama_ailab/finetuen_llama2_7b_lora.yaml [0,8] finetune
```

## 5. 权重合并
```python
python /home/ma-user/work/mindformers/mindformers/tools/transform_ckpt.py --src_ckpt_strategy /home/ma-user/work/mindformers/output/strategy/ --src_ckpt_dir /home/ma-user/work/mindformers/output/ckpt/ --dst_ckpt_dir /home/ma-user/work/ckpts/llama2-7b-lora/ --prefix llama2_7b_lora
```
## 6. 推理
```python
cd /home/work/mindformers/
python llama_test.py
```

## 7. Others
```python
# 同时杀死所有线程
ps -ef | grep "python run_mindformer.py" | grep -v grep | awk '{print $2}' | xargs kill -9
```