# >>> `run_predict.sh`文件
export RANK_TABLE_FILE=$1
export TYPE=$2
export MODEL=$3
export TOKENIZER=$4
export CHECKPOINT_PATH=$5
export INPUT_FILE=$6
# define variable
export RANK_SIZE=8
export START_RANK=0 # this server start rank
export END_RANK=8 # this server end rank
export LOCAL_DEFAULT_PATH="/home/ma-user/work/mindformers/output/"

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python /home/ma-user/work/mindformers/run_kg.py --device_id $i --type $TYPE --model $MODEL --tokenizer $TOKENIZER --checkpoint_path $CHECKPOINT_PATH --input_file $INPUT_FILE --use_parallel True &> /home/ma-user/work/mindformers/output/log/rank_$RANK_ID/run_kg.log &
done