#BIN=TRUST_modify
BIN=tc_challenge_1023
OUTPUT=${BIN}_result.log
sh run_ours_tuning.sh ./${BIN} network-datasets 7 $OUTPUT
sh run_ours_tuning.sh ./${BIN} Theory-datasets 7 $OUTPUT
sh run_ours_tuning.sh ./${BIN} mawi 7 $OUTPUT
sh run_ours_tuning.sh ./${BIN} graph500 7 $OUTPUT
sh run_ours_tuning.sh ./${BIN} new_datasets 7 $OUTPUT
sh run_ours_tuning.sh ./${BIN} others 7 $OUTPUT
sh run_ours_tuning.sh ./${BIN} genbank 7 $OUTPUT
