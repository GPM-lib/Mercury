BIN_NAME=${1}
DATASETS=${2}
echo "========"$DATASETS"========="
DATA_PATH=/data/lzh/graph_challenge_dataset/${DATASETS}
GPUID=${3}
OUTPUT=${4}

echo "========"$DATASETS"=========" >> $OUTPUT
# Loop over all items in the $DATA_PATH directory
for ITEM in $DATA_PATH/*
do
    # Check if the item is a directory
    if [ -d "$ITEM" ]
    then
	GRAPH=${ITEM}/graph
	NAME="${ITEM##*/}"
	#TIME=$(CUDA_VISIBLE_DEVICES=$GPUID $BIN_NAME $GRAPH | grep "Runtime " | grep -o -E '[0-9]+\.[0-9]+' | sed -n 1p)
    #TIME=$(mpirun -x  CUDA_VISIBLE_DEVICES=$GPUID -n 1 --bind-to numa $BIN_NAME $GRAPH  1 128 1024 256 0 0 | grep "Runtime " | grep -o -E '[0-9]+\.[0-9]+' | sed -n 1p)
    
    TIME=$(mpirun -x  CUDA_VISIBLE_DEVICES=$GPUID -n 1 --bind-to numa $BIN_NAME $GRAPH  1 1024 1024 32 0 0 | grep "Runtime " | grep -o -E '[0-9]+\.[0-9]+' | sed -n 1p)
    
    # echo "mpirun -x  CUDA_VISIBLE_DEVICES=$GPUID -n 1 --bind-to numa $BIN_NAME $GRAPH  1 128 1024 256 0 0"
    echo "$NAME  $TIME" >> $OUTPUT
    # ./graph_challenge/${DATASETS}/${NAME}.log
    echo "$NAME $TIME done!!!"
	# Print the directory name
        #echo $(basename "$ITEM")
    # break
    fi
done