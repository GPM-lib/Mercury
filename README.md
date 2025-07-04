
# Artifact for GraphChallenge'2024 champion&#x1F3C6;
> *MERCURY: Efficient Subgraph Matching on GPUs with Hybrid Scheduling, HPEC'2024* 

# 1. Dataset preparation
## 1.1. Download datasets.(amazon0302)

```
wget https://graphchallenge.s3.amazonaws.com/snap/amazon0302/amazon0302_adj.tsv
```

## 1.2. Convert tsv to mtx.

> amazon0302_adj.tsv -> amazon0302_adj.mtx

```
python3 ./datasets/tsv2mtx.py amazon0302_adj
```

## 1.3. Convert mtx to binary file.
>We adopt the same conversion script as [GraphMiner](https://github.com/chenxuhao/GraphAIBench). Thanks for Dr.XuhaoChen.

>For any dataset-related issues or requirements, please feel free to contact me at linzhiheng@ncic.ac.cn.
```
 amazon0302_adj.mtx -> amazon0302_adj/graph.meta.txt  
                    -> amazon0302_adj/graph.edge.bin 
                    -> amazon0302_adj/graph.vertex.bin
```

```
sh ./datasets/converter.sh amazon0302_adj.mtx amazon0302_adj
```


# 2. Compile implementation.
## 2.1 For single gpu triangle counting
```
make tc_challenge
```

## 2.2 For multi-gpu triangle counting
```
cd tc_multigpu && make tc_challenge_multigpu
```

## 2.3 For multi-gpu subgraph matching 
```
cd subgraph_matching && make sm_multigpu
```

# 3. Running code.
## 2.1 For single gpu triangle counting
>Usage: ./tc_challenge  <graph_path>

```
./tc_challenge  ~/data/cit-Patents/graph
```

## 2.2 For multi-gpu triangle counting
>Usage: mpirun -n <process_num> --bind-to numa ./tc_challenge_multigpu  <graph_path>

```
mpirun -n 4 --bind-to numa ./tc_challenge_multigpu ~/data/cit-Patents/graph
```

## 2.3 For multi-gpu subgraph matching 
>Usage: mpirun -n  <process_num> --bind-to numa  ./sm_multigpu <graph_path> <pattern_name>  
>Support Patterns(pattern_name): Pattern-Enumeration(P1,P2,P3,P4,P5,P6,P7,P8)

```
mpirun -n 4 --bind-to numa ./tc_challenge_multigpu ~/data/cit-Patents/graph P1
```
