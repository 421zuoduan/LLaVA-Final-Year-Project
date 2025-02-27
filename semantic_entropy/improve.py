import pickle
from semantic_entropy.uncertainty.uncertainty_measures.semantic_entropy import UncertaintyMeasures
import torch
import os
import json
from sklearn import metrics

def calculate_auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)

# 定义文件路径
base_path = 'playground/data/eval/pope/answers/300_10samples/'

# 读取calc_values.pkl
with open(base_path + 'calc_values.pkl', 'rb') as f:
    calc_data = pickle.load(f)

# 读取entropy_values.pkl
with open(base_path + 'entropy_values.pkl', 'rb') as f:
    entropy_data = pickle.load(f)    
    
# 将calc_data中的键值对转为独立变量
all_question_ids = calc_data['all_question_ids']
all_multi_responses = calc_data['all_multi_responses']
all_multi_sequences = calc_data['all_multi_sequences']
all_multi_log_liks = calc_data['all_multi_log_liks']
all_log_liks_agg = calc_data['all_log_liks_agg']
all_semantic_ids = calc_data['all_semantic_ids']
all_log_likelihood_per_semantic_id = calc_data['all_log_likelihood_per_semantic_id']

# 将entropy_data中的键值对转为独立变量
validation_is_false = entropy_data['validation_is_false']
all_regular_entropy = entropy_data['all_regular_entropy']
all_regular_entropy_rao = entropy_data['all_regular_entropy_rao']
all_semantic_entropy = entropy_data['all_semantic_entropy']
all_semantic_entropy_rao = entropy_data['all_semantic_entropy_rao']
all_cluster_assignment_entropy = entropy_data['all_cluster_assignment_entropy']
aurco_regular_entropy = entropy_data['aurco_regular_entropy']
aurco_regular_entropy_rao = entropy_data['aurco_regular_entropy_rao']
aurco_semantic_entropy = entropy_data['aurco_semantic_entropy']
aurco_semantic_entropy_rao = entropy_data['aurco_semantic_entropy_rao']
aurco_cluster_assignment_entropy = entropy_data['aurco_cluster_assignment_entropy']

uncertainty_computer = UncertaintyMeasures(question='test')

all_regular_entropy = []
all_regular_entropy_rao = []
all_semantic_entropy = []
all_semantic_entropy_rao = []


# print(f'len(all_multi_log_liks): {len(all_multi_log_liks)}')
# print(f'len(all_log_liks_agg): {len(all_log_liks_agg)}')
# # print(f'all_multi_log_liks[0].shape: {all_multi_log_liks[0].shape}')
# # print(f'all_log_liks_agg[0].shape: {all_log_liks_agg[0].shape}')
# print(f'all_multi_log_liks[0]: {all_multi_log_liks[0]}')
# print(f'all_log_liks_agg[0]: {all_log_liks_agg[0]}')

for idx, multi_log_liks, semantic_ids, multi_responses in zip(all_question_ids,  all_multi_log_liks, all_semantic_ids, all_multi_responses):
    # log_liks_agg = torch.tensor(multi_log_liks)
    # print(f'multi_log_liks: {multi_log_liks}')
    
    log_liks_agg = torch.tensor([torch.mean(log_lik) for log_lik in multi_log_liks])
    # print(f'log_liks_agg: {log_liks_agg}')
    
    
    ### Compute naive entropy
    regular_entropy = uncertainty_computer.predictive_entropy(log_liks_agg)
    regular_entropy_rao = uncertainty_computer.predictive_entropy_rao(log_liks_agg)
    all_regular_entropy.append(regular_entropy)
    all_regular_entropy_rao.append(regular_entropy_rao)
    print(f'regular_entropy: {regular_entropy}')
    print(f'regular_entropy_rao: {regular_entropy_rao}')
    
    
    ### Compute semantic entropy
    log_likelihood_per_semantic_id = uncertainty_computer.logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    all_log_likelihood_per_semantic_id.append(log_likelihood_per_semantic_id)
    print(f'log_likelihood_per_semantic_id: {log_likelihood_per_semantic_id}')
    
    # Compute semantic_entropy
    pe = uncertainty_computer.predictive_entropy(torch.tensor(log_likelihood_per_semantic_id))
    semantic_entropy = pe
    all_semantic_entropy.append(semantic_entropy)
    print(f'semantic_entropy: {semantic_entropy}')
    
    # Compute semantic_entropy_rao
    pe = uncertainty_computer.predictive_entropy_rao(torch.tensor(log_likelihood_per_semantic_id))
    semantic_entropy_rao = pe
    all_semantic_entropy_rao.append(semantic_entropy_rao)
    print(f'semantic_entropy_rao: {semantic_entropy_rao}')
    
### 计算 AUROC
print(f'validation_is_false: {validation_is_false}')

auroc_regular_entropy = calculate_auroc(validation_is_false, all_regular_entropy)
print(f'all_regular_entropy: {all_regular_entropy}')
print(f'auroc of regular_entropy: {auroc_regular_entropy}')

auroc_regular_entropy_rao = calculate_auroc(validation_is_false, all_regular_entropy_rao)
print(f'all_regular_entropy_rao: {all_regular_entropy_rao}')
print(f'auroc of regular_entropy_rao: {auroc_regular_entropy_rao}')

auroc_semantic_entropy = calculate_auroc(validation_is_false, all_semantic_entropy)
print(f'all_semantic_entropy: {all_semantic_entropy}')
print(f'auroc of semantic_entropy: {auroc_semantic_entropy}')

auroc_semantic_entropy_rao = calculate_auroc(validation_is_false, all_semantic_entropy_rao)
print(f'all_semantic_entropy_rao: {all_semantic_entropy_rao}')
print(f'auroc of semantic_entropy_rao: {auroc_semantic_entropy_rao}')

auroc_cluster_assignment_entropy = calculate_auroc(validation_is_false, all_cluster_assignment_entropy)
print(f'all_cluster_assignment_entropy: {all_cluster_assignment_entropy}')
print(f'auroc of cluster_assignment_entropy: {auroc_cluster_assignment_entropy}')
