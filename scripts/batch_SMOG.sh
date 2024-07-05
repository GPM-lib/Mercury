#BIN=TRUST_modify
BIN=SMOG
OUTPUT=${BIN}_result_1_0.1_8_216_1024_10.log
sh run_SMOG.sh ../SOTA/${BIN} network-datasets 7 $OUTPUT
sh run_SMOG.sh ../SOTA/${BIN} Theory-datasets 7 $OUTPUT
sh run_SMOG.sh ../SOTA/${BIN} mawi 7 $OUTPUT
sh run_SMOG.sh ../SOTA/${BIN} graph500 7 $OUTPUT
sh run_SMOG.sh ../SOTA/${BIN} new_datasets 7 $OUTPUT
sh run_SMOG.sh ../SOTA/${BIN} others 7 $OUTPUT
sh run_SMOG.sh ../SOTA/${BIN} genbank 7 $OUTPUT
