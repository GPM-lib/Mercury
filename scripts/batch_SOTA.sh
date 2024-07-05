#BIN=TRUST_modify
BIN=g2miner
OUTPUT=${BIN}_result.log
sh run_SOTA.sh ../SOTA/${BIN} network-datasets 7 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} Theory-datasets 7 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} mawi 7 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} graph500 7 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} new_datasets 7 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} others 7 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} genbank 6 $OUTPUT
