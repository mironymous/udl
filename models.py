import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
torch.manual_seed(0)

#TODO: Questions - whats up with the difference between n_train_samples and n_pred_samples?
# - why do they init with a MLE model?
# - How did they choose to regularize the KL divergence with the training size?
# - For the first task the mlp model is trained and then the coreset samples are selected. The baysian model are trained without the coreset samples

# parameter initionalisations
def weight_parameter(shape, init_weights=None):
    if init_weights is not None:
        initial = torch.tensor(init_weights)
    else:
        initial = torch.randn(shape) * 0.1
    return nn.Parameter(initial)

def bias_parameter(shape):
    initial = torch.ones(shape) * 0.1
    return nn.Parameter(initial)

def small_parameter(shape):
    initial = torch.ones(shape) * -6.0
    return nn.Parameter(initial)

def KL_of_gaussians(q_mean, q_logvar, p_mean, p_logvar):
    #double checked this it is correct. note: pull the exponent2 out of the log)
    return 0.5 * (p_logvar - q_logvar + (torch.exp(q_logvar) + (q_mean - p_mean)**2) / torch.exp(p_logvar) - 1)

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_train_samples = n_train_samples
        

    def train(self, X_train, y_train, task_id, n_epochs=1000, batch_size=512, lr=0.001):
        display_epoch = 5
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_train_samples = X_train.shape[0]
        if batch_size > n_train_samples:
            batch_size = n_train_samples
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                loss = self.calculate_loss(X_batch, y_batch, task_id)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if epoch % display_epoch == 0:
                #print truncated loss
                print(f'Epoch {epoch} Loss: {epoch_loss:.4f}')
        return losses
    
    def get_parameters(self):
        return self.params
        
    def predict(self, inputs, task_id):
        raise NotImplementedError

class MLP(NN):
    def __init__(self, input_dim, hidden_dims, output_dim, n_train_samples, lr=0.001):
        super().__init__(input_dim, hidden_dims, output_dim, n_train_samples)
        self.weights, self.biases, self.weights_last, self.biases_last = self.init_weights()
        self.params = nn.ParameterList([nn.ParameterList([self.weights, self.biases, self.weights_last, self.biases_last]), None])

    def _predict(self, inputs, task_id):
        activations = inputs
        for i in range(len(self.hidden_dims)):
            weights = self.weights[i]
            biases = self.biases[i]
            raw_activations = torch.matmul(activations, weights) + biases
            activations = torch.relu(raw_activations)
        logits = torch.matmul(activations, self.weights_last[task_id]) + self.biases_last[task_id]
        return logits

    def calculate_loss(self, X_batch, y_batch, task_id):
        logits = self._predict(X_batch, task_id)
        return nn.functional.cross_entropy(logits, y_batch)

    def predict_proba(self, X_test, task_id):
        return nn.functional.softmax(self._predict(X_test, task_id), dim=1)
    
    def predict(self, X_test, task_id):
        return torch.argmax(self.predict_proba(X_test, task_id), dim=1)

    def init_weights(self):
        self.layer_dims = deepcopy(self.hidden_dims)
        self.layer_dims.append(self.output_dim)
        self.layer_dims.insert(0, self.input_dim)
        n_layers = len(self.layer_dims) - 1
        
        weights = nn.ParameterList([])
        biases = nn.ParameterList([])
        weights_last = nn.ParameterList([]) #note that MLP is only called for the first task so this will only have one element
        biases_last = nn.ParameterList([])

        #Iterate over layers except the last
        for i in range(n_layers - 1):
            weights.append(weight_parameter((self.layer_dims[i], self.layer_dims[i+1])))
            biases.append(bias_parameter((self.layer_dims[i+1],)))
        #Last layer
        weights_last.append(weight_parameter((self.layer_dims[-2], self.layer_dims[-1])))
        biases_last.append(bias_parameter((self.layer_dims[-1],)))
        
        return weights, biases, weights_last, biases_last

class BNN(NN):
    def __init__(self, input_dim, hidden_dims, output_dim, training_size, n_train_samples=10, n_pred_samples = 100, previous_means=None, previous_log_variances=None, lr=0.001, prior_mean=torch.tensor(0), prior_var=torch.tensor(0), shared_head=False):
        #previous means is supplied as [weight_means, bias_means, weight_last_means, bias_last_means]
        #previous log variances is supplied equivalently
        #Note that variances are provided as log(variances)
        super().__init__(input_dim, hidden_dims, output_dim, n_train_samples)
        self.shared_head = shared_head
        means, variances = self.init_weights(previous_means, previous_log_variances)
        self.weight_means, self.bias_means, self.weight_last_means, self.bias_last_means = means
        self.weight_variances, self.bias_variances, self.weight_last_variances, self.bias_last_variances = variances
        self.params = nn.ParameterList([means, variances])
        self.training_size = training_size #used to regularize the KL divergence in ELBO loss

        means, variances = self.create_prior(previous_means, previous_log_variances, prior_mean, prior_var) 
        self.prior_weight_means, self.prior_bias_means, self.prior_weight_last_means, self.prior_bias_last_means = means
        self.prior_weight_variances, self.prior_bias_variances, self.prior_weight_last_variances, self.prior_bias_last_variances = variances

        self.n_layers = len(self.layer_dims) - 1
        self.n_train_samples = n_train_samples
        self.n_pred_samples = n_pred_samples

    def predict(self, inputs, task_id):
        return torch.argmax(self.predict_proba(inputs, task_id), dim=1)
    
    def predict_proba(self, X_test, task_id):
        sampled_logits = self._predict(X_test, task_id, self.n_pred_samples)
        probabilities = nn.functional.softmax(sampled_logits, dim=2)
        return torch.mean(probabilities, dim=0)    

    def _predict(self, inputs, task_id, n_samples):
        expanded_inputs = inputs.unsqueeze(0) #size: 1 x batch_size x input_dim = 1 x 64 x 784
        activations = expanded_inputs.repeat(n_samples, 1, 1) #size: n_pred_samples x batch_size x input_dim = 100 x 64 x 784
        for i in range(self.n_layers - 1):
            input_dim = self.layer_dims[i]
            output_dim = self.layer_dims[i+1]
            weight_epsilon = torch.randn(n_samples, input_dim, output_dim) #size: n_pred_samples x input_dim x output_dim
            bias_epsilon = torch.randn(n_samples, 1, output_dim) #size: n_pred_samples x 1 x output_dim
            #we use * 0.5 for the reparameterisation trick: taking the square root of the variance is the std
            weights = weight_epsilon * torch.exp(0.5 * self.weight_variances[i]) + self.weight_means[i]  
            biases = bias_epsilon * torch.exp(0.5 * self.bias_variances[i]) + self.bias_means[i]
            raw_activations = torch.matmul(activations, weights) + biases 
            activations = torch.relu(raw_activations) 
        input_dim = self.layer_dims[-2]
        output_dim = self.layer_dims[-1]
        weight_epsilon = torch.randn(n_samples, input_dim, output_dim)
        bias_epsilon = torch.randn(n_samples, 1, output_dim)

        weights = weight_epsilon * torch.exp(0.5 * self.weight_last_variances[task_id]) + self.weight_last_means[task_id]
        biases = bias_epsilon * torch.exp(0.5 * self.bias_last_variances[task_id]) + self.bias_last_means[task_id]
        #TODO: from the original code; check if this is correct 
        activations = activations.unsqueeze(3)
        weights = weights.unsqueeze(1)
        logits = torch.sum(activations * weights, dim=2) + biases
        return logits

    def log_likelihood_loss(self, inputs, targets, task_id):
        prediction = self._predict(inputs, task_id, self.n_train_samples)
        loss = 0
        for i in range(self.n_train_samples):
            loss += nn.functional.cross_entropy(prediction[i], targets, reduction="sum")
        return loss / self.n_train_samples

    def KL_loss(self, task_id):
        #For some reason the original code chooses to calculate KL for ALL prediction heads. So all of them are trained towards the prior at which they already are. I will just leave that out for now. 
        loss = 0
        for i in range(self.n_layers - 1):
            loss += torch.sum(KL_of_gaussians(self.weight_means[i], self.weight_variances[i], self.prior_weight_means[i], self.prior_weight_variances[i]))
            loss += torch.sum(KL_of_gaussians(self.bias_means[i], self.bias_variances[i], self.prior_bias_means[i], self.prior_bias_variances[i]))
        
        loss += torch.sum(KL_of_gaussians(self.weight_last_means[task_id], self.weight_last_variances[task_id], self.prior_weight_last_means[task_id], self.prior_weight_last_variances[task_id]))
        loss += torch.sum(KL_of_gaussians(self.bias_last_means[task_id], self.bias_last_variances[task_id], self.prior_bias_last_means[task_id], self.prior_bias_last_variances[task_id]))
        #original implementation uses KL for all prediction heads
        #that will lead to strange behavior for coreset scenario where the model is trained for the next task and the KL term trains there previous tasks prediction head towards its last state
        return loss
    
    def calculate_loss(self, inputs, targets, task_id):
        #compute elboloss and regularize KL divergence by scaling with the training size (as in the original implementation)
        log_likelihood_loss = self.log_likelihood_loss(inputs, targets, task_id)
        kl_loss = self.KL_loss(task_id)
        return kl_loss / inputs.shape[0] + log_likelihood_loss


    def init_weights(self, previous_means, previous_log_variances):
        #previous means is supplied as [weight_means, bias_means, weight_last_means, bias_last_means]
        #previous log variances is supplied equivalently
        
        #Note that the first task is trained to ML so there will be no variance but means.
        #note that this will make use of and change the previous BNNs parameters
        weight_means = nn.ParameterList([])
        bias_means = nn.ParameterList([])
        weight_last_means = nn.ParameterList([])
        bias_last_means = nn.ParameterList([])

        weight_variances = nn.ParameterList([])
        bias_variances = nn.ParameterList([])
        weight_last_variances = nn.ParameterList([])
        bias_last_variances = nn.ParameterList([])

        self.layer_dims = deepcopy(self.hidden_dims)
        self.layer_dims.append(self.output_dim)
        self.layer_dims.insert(0, self.input_dim)
        n_layers = len(self.layer_dims) - 1

        for i in range(n_layers - 1):
            if previous_means is None:
                weight_mean = weight_parameter((self.layer_dims[i], self.layer_dims[i+1]))
                bias_mean = bias_parameter((self.layer_dims[i+1],))
                weight_variance = small_parameter((self.layer_dims[i], self.layer_dims[i+1])) 
                bias_variance = small_parameter((self.layer_dims[i+1],))
            else:
                weight_mean = nn.Parameter(previous_means[0][i])
                bias_mean = nn.Parameter(previous_means[1][i])
                if(previous_log_variances is None):
                    weight_variance = small_parameter((self.layer_dims[i], self.layer_dims[i+1]))
                    bias_variance = small_parameter((self.layer_dims[i+1],))
                else:
                    weight_variance = nn.Parameter(previous_log_variances[0][i])
                    bias_variance = nn.Parameter(previous_log_variances[1][i])
            weight_means.append(weight_mean)
            bias_means.append(bias_mean)
            weight_variances.append(weight_variance)
            bias_variances.append(bias_variance)
        
        if(previous_log_variances is not None and previous_means is not None):
            n_previous_heads = len(previous_means[2])
            for i in range(n_previous_heads):
                weight_last_means.append(nn.Parameter(previous_means[2][i]))
                bias_last_means.append(nn.Parameter(previous_means[3][i]))
                weight_last_variances.append(nn.Parameter(previous_log_variances[2][i]))
                bias_last_variances.append(nn.Parameter(previous_log_variances[3][i]))
        
        if(not self.shared_head):
            if(previous_log_variances is None and previous_means is not None):
                weight_last_means.append(nn.Parameter(previous_means[2][0]))
                bias_last_means.append(nn.Parameter(previous_means[3][0]))
            else:
                weight_last_means.append(weight_parameter((self.layer_dims[-2], self.layer_dims[-1])))
                bias_last_means.append(bias_parameter((self.layer_dims[-1],)))
            weight_last_variances.append(small_parameter((self.layer_dims[-2], self.layer_dims[-1])))
            bias_last_variances.append(small_parameter((self.layer_dims[-1],)))
        else:
            if(previous_log_variances is None and previous_means is not None):
                weight_last_means.append(nn.Parameter(previous_means[2][0]))
                bias_last_means.append(nn.Parameter(previous_means[3][0]))
                weight_last_variances.append(small_parameter((self.layer_dims[-2], self.layer_dims[-1])))
                bias_last_variances.append(small_parameter((self.layer_dims[-1],)))

        return [weight_means, bias_means, weight_last_means, bias_last_means], [weight_variances, bias_variances, weight_last_variances, bias_last_variances]

    def create_prior(self, previous_means, previous_variances, prior_mean, prior_var):
        #previous means is supplied as [weight_means, bias_means, weight_last_means, bias_last_means]
        #previous log variances is supplied equivalently
        self.layer_dims = deepcopy(self.hidden_dims)
        self.layer_dims.append(self.output_dim)
        self.layer_dims.insert(0, self.input_dim)
        n_layers = len(self.layer_dims) - 1
        weight_means = []
        bias_means = []
        weight_last_means = []
        bias_last_means = []
        weight_variances = []
        bias_variances = []
        weight_last_variances = []
        bias_last_variances = []
        
        if(previous_variances is not None):
        #if the previous model has been  trained with VI already
        #note that when initialising with MLP weights we still use prior 0 for all weights
            for i in range(n_layers - 1):
                weight_means.append(previous_means[0][i].detach().clone())
                bias_means.append(previous_means[1][i].detach().clone())
                weight_variances.append(previous_variances[0][i].detach().clone())
                bias_variances.append(previous_variances[1][i].detach().clone())

            n_previous_heads = len(previous_means[2])
            for i in range(n_previous_heads):
                weight_last_means.append(previous_means[2][i].detach().clone())
                bias_last_means.append(previous_means[3][i].detach().clone())
                weight_last_variances.append(previous_variances[2][i].detach().clone())
                bias_last_variances.append(previous_variances[3][i].detach().clone())
        else:
            for i in range(n_layers - 1):
                weight_means.append(prior_mean)
                bias_means.append(prior_mean)
                weight_variances.append(prior_var)
                bias_variances.append(prior_var)
        if(not self.shared_head):
            weight_last_means.append(prior_mean)
            bias_last_means.append(prior_mean)
            weight_last_variances.append(prior_var)
            bias_last_variances.append(prior_var)
        else:
            if(previous_variances is None):
                weight_last_means.append(prior_mean)
                bias_last_means.append(prior_mean)
                weight_last_variances.append(prior_var)
                bias_last_variances.append(prior_var)
        
        return [weight_means, bias_means, weight_last_means, bias_last_means], [weight_variances, bias_variances, weight_last_variances, bias_last_variances]

class SingleHeadMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.backbone_dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(self.backbone_dims) - 1):
            self.layers.append(nn.Linear(self.backbone_dims[i], self.backbone_dims[i+1]))
        self.head = nn.Linear(self.backbone_dims[-1], output_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)
    
    def train(self, X_train_by_task, y_train_by_task, n_epochs, batch_size, lr=0.001):
        #merge all tasks into one dataset
        X_train = torch.cat(X_train_by_task, dim=0)
        y_train = torch.cat(y_train_by_task, dim=0)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        display_epoch = 5
        result = []
        for epoch in range(n_epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = torch.nn.functional.cross_entropy(outputs, torch.argmax(y_batch, dim=1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            result.append(epoch_loss)
            if epoch % display_epoch == 0:
                print(f'Epoch {epoch} Loss: {epoch_loss:.4f}')
        return result
    
    def evaluate(self, X_test_by_task, y_test_by_task):
        accuracies = []
        for i in range(len(X_test_by_task)):
            X_test, y_test = X_test_by_task[i], y_test_by_task[i]
            predictions = self(X_test)
            prediction = torch.argmax(predictions, dim=1)
            y = torch.argmax(y_test, dim=1)
            accuracy = torch.sum(prediction == y).item() / y.shape[0]
            accuracies.append(accuracy)
        return accuracies
    

class MultiheadMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n_heads):
        super().__init__()
        self.backbone_dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(self.backbone_dims) - 1):
            self.layers.append(nn.Linear(self.backbone_dims[i], self.backbone_dims[i+1]))
        self.heads = nn.ModuleList([nn.Linear(self.backbone_dims[-1], output_dim) for _ in range(n_heads)])
    
    def forward(self, x, head):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.heads[head](x)
    
    def train(self, X_train_by_task, y_train_by_task, n_epochs, batch_size, lr=0.001):
        dataloaders = []
        for X_train, y_train in zip(X_train_by_task, y_train_by_task):
            dataset = TensorDataset(X_train, y_train)
            dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        display_epoch = 5
        result = []
        num_batches_by_task = [len(loader) for loader in dataloaders]
        max_batches = max(num_batches_by_task)
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_id in range(max_batches):
                batch_loss = 0
                for task_id, dataloader in enumerate(dataloaders):
                    batch_X, batch_y = next(iter(dataloader))
                    optimizer.zero_grad()
                    outputs = self(batch_X, task_id)
                    loss = torch.nn.functional.cross_entropy(outputs, torch.argmax(batch_y, dim=1))
                    batch_loss += loss
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            result.append(epoch_loss)

            if epoch % display_epoch == 0:
                print(f'Epoch {epoch} Loss: {epoch_loss:.4f}')
        return result
    
    def evaluate(self, X_test_by_task, y_test_by_task):
        accuracies = []
        for i in range(len(X_test_by_task)):
            X_test, y_test = X_test_by_task[i], y_test_by_task[i]
            predictions = self(X_test, i)
            prediction = torch.argmax(predictions, dim=1)
            y = torch.argmax(y_test, dim=1)
            accuracy = torch.sum(prediction == y).item() / y.shape[0]
            accuracies.append(accuracy)
        return accuracies
    

