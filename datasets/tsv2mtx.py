import networkx as nx
import pandas as pd
import sys


csvfile = sys.argv[1]

print("Reading file ...")
# Load TSV file
df = pd.read_csv(csvfile+".tsv", sep='\t', header=None)

# Create graph from DataFrame
G = nx.from_pandas_edgelist(df, source=0, target=1)

# Calculate number of vertices
#num_vertices = G.number_of_nodes()

# Calculate number of edges
#num_edges = G.number_of_edges()

#print(f'Total number of vertices: {num_vertices}')
#print(f'Total number of edges: {num_edges}')

outputfile = open(csvfile+".mtx", "w")
print("Writing file ...")
outputfile.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
outputfile.write("%-------------------------------------------------------------------------------\n")
outputfile.write(str(G.number_of_nodes()) + " " + str(G.number_of_nodes()) + " " + str(G.number_of_edges()) + "\n")
for edges in G.edges:
	x = edges[0]
	y = edges[1]
	outputfile.write(str(x) + " " + str(y) + "\n")
outputfile.close()
print("Complete!")


