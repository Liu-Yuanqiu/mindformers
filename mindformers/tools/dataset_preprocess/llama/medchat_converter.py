"""
fastchat stanford alpaca data convert tools.
"""

import argparse

import json

import pathlib

def main(args_param):
    data_path = pathlib.Path(args_param.data_path)
    with data_path.open() as f:
        data = json.load(f)
    source = [d["instruction"]+d["input"]+"<MedChat>"+d["output"] for d in data]

    new_data = []
    cnt = 1

    for s in source:
        s = s.replace("<MedChat>", "<User>")
        ss = s.split("<User>")[1:]
        conversations = []
        for i, sss in enumerate(ss):
            if i%2==0:
                conversations.append({
                    "from": "User",
                    "value": sss
                })
            else:
                conversations.append({
                    "from": "MedChat",
                    "value": sss
                })
        new_data.append({
                "id": str(cnt),
                "conversations": conversations
            })

        cnt += 1

    json.dump(new_data, open(args_param.output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca-data.json")
    parser.add_argument(
        "--output_path", type=str, default="alpaca-data-conversation.json"
    )
    args = parser.parse_args()
    main(args)
