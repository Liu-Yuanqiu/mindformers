from mindformers import Trainer
import mindspore as ms
import os
import json

# ms.set_context(mode=0)
# cls_trainer = Trainer(task="text_generation", # 已支持的任务名
#                       model="llama_7b",) # 已支持的模型名

# 根据alpaca数据集的prompt模板，在instruction处填入自定义指令
# input_data = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:".format("Tell me about DLUT.")

# 方式1： 传入lora微调后的权重进行推理
# lora_ckpt = "/home/ma-user/work/ckpts/ms-llama-7b.ckpt"
# predict_result = cls_trainer.predict(input_data=input_data,
                                    #  predict_checkpoint=lora_ckpt)

# 方式2： 从obs下载训练好的权重进行推理
# predict_result = cls_trainer.predict(input_data=input_data)
# print(predict_result)
# print(predict_result[0]["text_generation_text"][0])

# output:
# Alpacas are a species of South American camelid. They are domesticated animals that are raised for their wool, meat, and milk. Alpacas are gentle, docile animals that are very friendly and easy to care for. They are also very intelligent and can be trained to perform certain tasks. Alpacas are very social animals and live in herds of up to 20 individuals. They are also very vocal and can make a variety of sounds, including a loud, high-pitched bark.

# data = []
# for i in range(8):
#     d = json.load(open(os.path.join("/home/ma-user/work/mindformers/output/log", "rank_"+str(i), "sens_entity_1017.json"), "r", encoding="utf-8"))
#     print(len(d))
#     data += d
# print(len(data))
# json.dump(data, open("/home/ma-user/work/data/kg/sens_entity_1017.json", "w", encoding="utf-8"), ensure_ascii=False)

data = json.load(open("/home/ma-user/work/data/kg/sens_kg_1017.json", "r", encoding="utf-8"))
print(data[0])
# sen_num = 0
# entity_num = 0
# entity_set = set()
# for d in data:
#     sen_num += 1
#     out = d["output"]
#     for e in out.split(","):
#         entity_num += 1
#         entity_set.add(e.strip())
# print(sen_num)
# print(entity_num)
# print(len(entity_set))