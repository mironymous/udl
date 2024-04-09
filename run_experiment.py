
from datasets import SplitNotMNIST, SplitMnist, PermutedMnist
import torch
import coreset
import vcl
import baseline
import pickle
import os
import sys

experiments_by_id = [{'dataset': dataset, 'method': method, 'seed': seed} for dataset in ['split_notmnist', 'split_mnist', 'permuted_mnist'] for method in ['vcl', 'random_coreset', 'k_center', 'baseline'] for seed in range(5)]

def run_experiment(dataset, method, seed):
    print("Running experiment with dataset: ", dataset, "method: ", method, " seed: ", seed)
    torch.manual_seed(seed)
    batch_size = 512
    if dataset == 'split_notmnist':
        data_class = SplitNotMNIST()
        shared_head = False
        hidden_dimensions = [150, 150, 150, 150]
        n_epochs = 100
    elif dataset == 'split_mnist':
        data_class = SplitMnist()
        shared_head = False
        hidden_dimensions = [256, 256]
        n_epochs = 120
    elif dataset == 'permuted_mnist':
        data_class = PermutedMnist(10)
        shared_head = True
        hidden_dimensions = [100, 100]
        n_epochs = 100
    else:
        raise ValueError('Invalid dataset')
    
    if(method == 'vcl'):
        result, training_times = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, 0, batch_size, shared_head)
    elif(method == 'random_coreset'):
        result, training_times = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.random_coreset, 40, batch_size, shared_head)
    elif(method == 'k_center'):
        result, training_times = vcl.run_vcl(hidden_dimensions, n_epochs, data_class, coreset.k_center, 40, batch_size, shared_head)
    elif(method == 'baseline'):
        result, training_times = baseline.run_baseline(hidden_dimensions, n_epochs, data_class, batch_size, shared_head)
    
    print("Result: ", result)
    print("Training times: ", training_times)

    result_path = os.path.join('results', dataset, method, str(seed))
    os.makedirs(result_path, exist_ok=True)
    #store results and training times
    with open(os.path.join(result_path, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    with open(os.path.join(result_path, 'training_times.pkl'), 'wb') as f:
        pickle.dump(training_times, f)

if __name__ == '__main__':
    id = int(sys.argv[1])
    experiment = experiments_by_id[id]
    run_experiment(experiment['dataset'], experiment['method'], experiment['seed'])
    
    
