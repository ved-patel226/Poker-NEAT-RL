"""
speciation: δ = (c1 × E / N) + (c2 × D / N) + (c3 × W̄)
E = num of exess genes
D = num of disjoint genes
N = num of genes in the larger genome
W̄ = average weight diff in matching genes
c1, c2, c3 = constants, YOU HAVE TO TUNE TS (normally: 1.0, 1.0, 0.4)


If δ < threshold, its the same species

"""
