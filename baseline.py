
from models import MultiheadMLP, SingleHeadMLP
import time

def run_baseline(hidden_dims, n_epochs, data_class, batch_size=264, shared_head=True):
    input_dim, out_dim = data_class.get_dims()
    task_accuracies = []
    X_train_by_task, y_train_by_task = [], []
    X_test_by_task, y_test_by_task = [], []
    training_times = []
    
    for task_id in range(data_class.n_tasks):
        print('Task ', task_id)
        X_train, y_train, X_test, y_test = data_class.next_task()
        X_train_by_task.append(X_train)
        y_train_by_task.append(y_train)
        X_test_by_task.append(X_test)
        y_test_by_task.append(y_test)

        if(shared_head):
            model = SingleHeadMLP(input_dim, hidden_dims, out_dim)
        else:
            model = MultiheadMLP(input_dim, hidden_dims, out_dim, data_class.n_tasks)
        start = time.time()
        model.train(X_train_by_task, y_train_by_task, n_epochs, batch_size)
        end = time.time()
        training_times.append(end-start)
        accuracies = model.evaluate(X_test_by_task, y_test_by_task)
        task_accuracies.append(accuracies)
    return task_accuracies, training_times