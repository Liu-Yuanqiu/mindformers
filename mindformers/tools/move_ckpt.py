import os
import shutil
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_pre_name', type=str, default='llama2')
    parser.add_argument('--ckpt_post_name', type=str, default='llama2')
    args = parser.parse_args()

    def get_ckpt_name(rank):
        return args.ckpt_pre_name+str(rank)+args.ckpt_post_name+".ckpt"
    
    OUTPUT_PATH = "/home/ma-user/work/mindformers/output/"
    CKPT_PATH = "/home/ma-user/work/mindformers/output/checkpoint/"
    if os.path.exists(os.path.join(CKPT_PATH, "rank_0", get_ckpt_name(0))):
        for i in range(8):
            NEW_PATH = os.path.join(OUTPUT_PATH, "ckpt", "rank_"+str(i))
            os.makedirs(NEW_PATH)
            shutil.copy(os.path.join(CKPT_PATH, "rank_"+str(i), get_ckpt_name(i)), \
                        os.path.join(NEW_PATH, get_ckpt_name(i)))
    else:
        print("目标文件不存在！")