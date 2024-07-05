make clean && make -j20 && mpirun -x  CUDA_VISIBLE_DEVICES=4,5,6,7  -n  4  --bind-to numa   ./tc_challenge_multigpu ~/data/cit-Patents/graph
