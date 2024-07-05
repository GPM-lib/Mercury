BIN=TRUST_modify

OUTPUT=${BIN}_result.log
sh run_SOTA.sh ../SOTA/${BIN} network-datasets 6 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} Theory-datasets 6 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} mawi 6 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} graph500 6 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} new_datasets 6 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} others 6 $OUTPUT
sh run_SOTA.sh ../SOTA/${BIN} genbank 6 $OUTPUT

