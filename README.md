# AAAI-2024: Identifying Guarantors of War Veterans Using Robust-SEAL: A Case of the Korean War

## Abstract

Many veterans suffer from mental and physical problems. To reward their sacrifice, many countries provide veterans with various benefits. However, quite a few veterans have failed to prove their status due to loss of military records. Thus, some governments allow verification of the veterans through "buddy statements" from the people who can vouch for the buddy's participation in the war. However, it is still challenging for veterans to find guarantors who can write buddy statements as many war records have been lost and their accuracy varies. With this background, this study utilizes historical war records to increase the pool of potential guarantors for the buddy statements. We construct a combined operation network among troops in the war which might include missing edges data and perturbations on attributes of the troop. Then we predict missing linkages among the troops that might have interacted together in the war. We propose a Robust-SEAL (learning from Subgraphs, Embeddings, and Attributes for Link prediction) by combining two GNN architectures, robust GCN which considers the uncertainty of node attributes with a probabilistic approach, and SEAL which helps to improve expressive power of the GNN with a labeling trick. Our proposed approach was applied to Korean War data with perturbations. For experimentations, we hid some interactions and found that Robust-SEAL restores missing interactions better than baselines.

## Environment

Google colab pro was used
- CPU = Intel(R) Xeon(R) CPU @ 2.30GHz
- GPU = NVIDIA Tesla T4
- RAM = 25.51GB

## Requirements

- Python == 3.8.10
- torch == 1.12.0+cu113
- torch_geometric == 2.0.4
- numpy == 1.22.4
- scipy == 1.7.3
- sklearn == 1.0.2

## Dataset

A dataset was extracted from The War Memorial of the Korea(https://www.warmemo.or.kr/front/militaryInfo/warDeadSearch.do) 
and Cho et al. (2017).
Nodes represent Army units that participated in operations in the war.
Edges indicate the number of combined operations that the two units participated in together.

features.csv includes node features in the input network.
- node: index of the node
- Division_i : whether the node belongs to the division i
- Regiment_i : whether the node belongs to the regiment i
- period_k : whether the node participated in the period k
- KIA : the number of killed in action of the node

edge_list.csv represents the edge list with weights.
- Source: index of the node
- Target: index of the node
- Value: the number of combined operations that the two units participated in together
