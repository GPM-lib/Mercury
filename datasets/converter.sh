DATA=$1
OUTPUT_DIR=$2

mkdir $OUTPUT_DIR
./converter  mtx $DATA  $OUTPUT_DIR/graph 1 0 0 0
#./converter_vidfrom0  mtx $DATA  $OUTPUT_DIR/graph 1 0 0 0
