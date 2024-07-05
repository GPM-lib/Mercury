#BIN=TRUST_modify
BIN=HINDEX
OUTPUT=${BIN}_result.log1111111
sh run_HINDEX.sh ../SOTA/${BIN} network-datasets 7 $OUTPUT
sh run_HINDEX.sh ../SOTA/${BIN} Theory-datasets 7 $OUTPUT
sh run_HINDEX.sh ../SOTA/${BIN} mawi 7 $OUTPUT
sh run_HINDEX.sh ../SOTA/${BIN} graph500 7 $OUTPUT
sh run_HINDEX.sh ../SOTA/${BIN} new_datasets 7 $OUTPUT
sh run_HINDEX.sh ../SOTA/${BIN} others 7 $OUTPUT
sh run_HINDEX.sh ../SOTA/${BIN} genbank 7 $OUTPUT
