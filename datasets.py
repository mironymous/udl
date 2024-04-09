import torchvision
import torch
from torchvision import transforms, datasets
import os
from torchvision.datasets.utils import download_url
from PIL import Image

class SplitMnist():
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),     

        ])
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        #flatten the image
        self.X_train = torch.stack([sample[0].view(-1) for sample in mnist_train])
        self.X_test = torch.stack([sample[0].view(-1) for sample in mnist_test])
        self.train_label = torch.tensor([sample[1] for sample in mnist_train])
        self.test_label = torch.tensor([sample[1] for sample in mnist_test])

        self.labels_0 = [0, 2, 4, 6, 8]
        self.labels_1 = [1, 3, 5, 7, 9]

        self.current_task = 0
        self.n_tasks = len(self.labels_0)

    def get_dims(self):
        # Return the number of features and number of classes
        return 784, 2

    def next_task(self):
        if self.current_task >= self.n_tasks:
            raise Exception('All tasks completed already')
        else:
            negative_ids_train = torch.nonzero(self.train_label == self.labels_0[self.current_task]).squeeze(1)
            positive_ids_train = torch.nonzero(self.train_label == self.labels_1[self.current_task]).squeeze(1)
            X_train = torch.cat((self.X_train[negative_ids_train], self.X_train[positive_ids_train]), dim=0)

            y_train = torch.cat((torch.ones(negative_ids_train.shape[0], dtype=torch.float).unsqueeze(1),
                                      torch.zeros(positive_ids_train.shape[0], dtype=torch.float).unsqueeze(1)), dim=0)
            y_train = torch.cat((y_train, 1 - y_train), dim=1)
            #y_train[:,0] contains the labels and y_train[:,1] contains the complementary labels
            negative_ids_test = torch.nonzero(self.test_label == self.labels_0[self.current_task]).squeeze(1)
            positive_ids_test = torch.nonzero(self.test_label == self.labels_1[self.current_task]).squeeze(1)
            X_test = torch.cat((self.X_test[negative_ids_test], self.X_test[positive_ids_test]), dim=0)

            y_test = torch.cat((torch.ones(negative_ids_test.shape[0], dtype=torch.float).unsqueeze(1),
                                     torch.zeros(positive_ids_test.shape[0], dtype=torch.float).unsqueeze(1)), dim=0)
            y_test = torch.cat((y_test, 1 - y_test), dim=1)

            self.current_task += 1

            return X_train, y_train, X_test, y_test
    
    def reset(self):
        self.current_task = 0
    

class PermutedMnist():
    def __init__(self, n_tasks=10):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        #flatten the image
        self.X_train = torch.stack([sample[0].view(-1) for sample in mnist_train])
        self.X_test = torch.stack([sample[0].view(-1) for sample in mnist_test])
        self.y_train = torch.tensor([sample[1] for sample in mnist_train])
        self.y_test = torch.tensor([sample[1] for sample in mnist_test])

        self.n_tasks = n_tasks
        self.current_task = 0

        self.permutations = []
        for _ in range(n_tasks):
            self.permutations.append(torch.randperm(784))

    def get_dims(self):
        return 784, 10

    def next_task(self):
        if self.current_task >= self.n_tasks:
            raise Exception('All tasks are already completed')
        else:
            X_train = self.X_train.detach().clone()[:, self.permutations[self.current_task]]
            y_train = self.y_train.detach().clone()
            X_test = self.X_test.detach().clone()[:, self.permutations[self.current_task]]
            y_test = self.y_test.detach().clone()
            y_train = torch.nn.functional.one_hot(y_train, num_classes=10).float()
            y_test = torch.nn.functional.one_hot(y_test, num_classes=10).float()
            
            self.current_task += 1
        return X_train, y_train, X_test, y_test
    
    def reset(self):
        self.current_task = 0



class SplitNotMNIST():
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),     
        ])

        data_path = 'data/notMNIST/'
        url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz'

        if not os.path.exists(data_path):
            os.makedirs(data_path)
            download_url(url, root=data_path, filename='notMNIST_small.tar.gz', md5=None)

        if not os.path.exists(os.path.join(data_path, 'notMNIST_small')):
            import tarfile
            with tarfile.open(os.path.join(data_path, 'notMNIST_small.tar.gz'), 'r:gz') as tar:
                tar.extractall(path=data_path)
        
        self.verify_images(os.path.join(data_path, 'notMNIST_small'))

        notmnist_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'notMNIST_small'), transform=transform)
        train_size = int(0.8 * len(notmnist_dataset))
        test_size = len(notmnist_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(notmnist_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

        self.X_train = torch.stack([sample[0].view(-1) for sample in train_dataset])
        self.X_test = torch.stack([sample[0].view(-1) for sample in test_dataset])
        self.train_label = torch.tensor([sample[1] for sample in train_dataset])
        self.test_label = torch.tensor([sample[1] for sample in test_dataset])

        self.labels_0 = [0, 1, 2, 3, 4]
        self.labels_1 = [5, 6, 7, 8, 9]

        self.current_task = 0
        self.n_tasks = len(self.labels_0)

    

    def verify_images(self, root_dir):
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                if filename.lower().endswith(('.png', )):
                    file_path = os.path.join(root, filename)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                    except (IOError, SyntaxError) as e:
                        print(f'Removing Bad file: {file_path} - {e}')
                        #remove bad file
                        os.remove(file_path)
                    


    def get_dims(self):
        # Return the number of features and number of classes
        return 2352, 2

    def next_task(self):
        if self.current_task >= self.n_tasks:
            raise Exception('All tasks completed already')
        else:
            negative_ids_train = torch.nonzero(self.train_label == self.labels_0[self.current_task]).squeeze(1)
            positive_ids_train = torch.nonzero(self.train_label == self.labels_1[self.current_task]).squeeze(1)
            X_train = torch.cat((self.X_train[negative_ids_train], self.X_train[positive_ids_train]), dim=0)

            y_train = torch.cat((torch.ones(negative_ids_train.shape[0], dtype=torch.float).unsqueeze(1),
                                      torch.zeros(positive_ids_train.shape[0], dtype=torch.float).unsqueeze(1)), dim=0)
            y_train = torch.cat((y_train, 1 - y_train), dim=1)
            #y_train[:,0] contains the labels and y_train[:,1] contains the complementary labels
            negative_ids_test = torch.nonzero(self.test_label == self.labels_0[self.current_task]).squeeze(1)
            positive_ids_test = torch.nonzero(self.test_label == self.labels_1[self.current_task]).squeeze(1)
            X_test = torch.cat((self.X_test[negative_ids_test], self.X_test[positive_ids_test]), dim=0)

            y_test = torch.cat((torch.ones(negative_ids_test.shape[0], dtype=torch.float).unsqueeze(1),
                                     torch.zeros(positive_ids_test.shape[0], dtype=torch.float).unsqueeze(1)), dim=0)
            y_test = torch.cat((y_test, 1 - y_test), dim=1)

            self.current_task += 1

            return X_train, y_train, X_test, y_test
    
    def reset(self):
        self.current_task = 0