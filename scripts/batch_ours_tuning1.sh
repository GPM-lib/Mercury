#BIN=TRUST_modify
BIN=tc_challenge_1031
OUTPUT=${BIN}_result.log
sh run_ours_tuning.sh ./${BIN} network-datasets 6 $OUTPUT
sh run_ours_tuning.sh ./${BIN} Theory-datasets 6 $OUTPUT
sh run_ours_tuning.sh ./${BIN} mawi 6 $OUTPUT
sh run_ours_tuning.sh ./${BIN} graph500 6 $OUTPUT
sh run_ours_tuning.sh ./${BIN} new_datasets 6 $OUTPUT
sh run_ours_tuning.sh ./${BIN} others 6 $OUTPUT
sh run_ours_tuning.sh ./${BIN} genbank 6 $OUTPUT
