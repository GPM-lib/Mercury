DATASETS=${1}
echo "========"$DATASETS"========="
DATA_PATH=/data/lzh/graph_challenge_dataset/${DATASETS}
GPUID=6
BIN_NAME=tc_challenge
OUTPUT=our_result.log

echo "========"$DATASETS"=========" >> $OUTPUT
# Loop over all items in the $DATA_PATH directory
for ITEM in $DATA_PATH/*
do
    # Check if the item is a directory
    if [ -d "$ITEM" ]
    then
	GRAPH=${ITEM}/graph
	NAME="${ITEM##*/}"
	TIME=$(CUDA_VISIBLE_DEVICES=$GPUID ./$BIN_NAME $GRAPH | grep "Runtime " | grep -o -E '[0-9]+\.[0-9]+' | sed -n 1p)
    echo "$NAME  $TIME" >> $OUTPUT
    # ./graph_challenge/${DATASETS}/${NAME}.log
    echo "$NAME $TIME done!!!"
	# Print the directory name
        #echo $(basename "$ITEM")
    #break
    fi
done