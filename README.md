# AAAI-2024: Identifying Guarantors of War Veterans Using Robust-SEAL: A Case of the Korean War

## Abstract

Most countries provide veterans with various benefits to re-ward their sacrifice. Unfortunately, many veterans have failed to prove their status due to loss of military records. Thus, some governments allow the verification of those veterans through "buddy statements" obtained from the people who can vouch for the buddy's participation in the war. However, it is still challenging for veterans to find guarantors directly. With this background, we suggest to utilizing historical war records of combined operations to increase the pool of poten-tial guarantors for the buddy statements. However, a com-bined operation network among troops can have missing edg-es and perturbations on attributes of the troop due to inaccu-rate information. In order to predict both missing linkages among the troops that might have interacted together in the war and the potential interaction including knowledge flow even without participating in the same combined operations, we propose Robust-SEAL (learning from Subgraphs, Em-beddings, and Attributes for Link prediction). It combines two Graph Neural Network (GNN) architectures: robust Graph Convolutional Network which considers the uncertain-ty of node attributes with a probabilistic approach, and SEAL which improves the expressive power of the GNN with a la-beling trick. Our proposed approach was applied to Korean War data with perturbations. For experimentations, we hid some actual interactions and found that Robust-SEAL re-stores missing interactions better than other GNN-based baselines.

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
