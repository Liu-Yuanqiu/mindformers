import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_name', default='llama2')
    args = parser.parse_args()

    OUTPUT_PATH = "/home/ma-user/work/mindformers/output/"
    CKPT_PATH = "/home/ma-user/work/mindformers/output/checkpoint/"
    if os.path.exists(os.path.join(CKPT_PATH, "rank_0", args.ckpt_name + ".ckpt")):
        for i in range(8):
            NEW_PATH = os.path.join(OUTPUT_PATH, "ckpt", "rank_"+str(i))
            os.mkdir(NEW_PATH)
            shutil.copy(os.path.join(CKPT_PATH, "rank_"+str(i), args.ckpt_name + ".ckpt"), \
                        os.path.join(NEW_PATH, args.ckpt_name + ".ckpt"))
    else:
        print("目标文件不存在！")
