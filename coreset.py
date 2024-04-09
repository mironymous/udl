import torch

def random_coreset(X_cs, y_cs, X, y, size):
    ids = torch.randperm(X.size(0))[:size]
    X_cs.append(X[ids])
    y_cs.append(y[ids])
    not_ids = torch.tensor([i for i in range(X.size(0)) if i not in ids])
    X = torch.index_select(X, 0, not_ids)
    y = torch.index_select(y, 0, not_ids)
    return X_cs, y_cs, X, y

def k_center(X_cs, y_cs, X, y, size):
    distances = torch.full((X.size(0),), float('inf'))
    id = 0
    distances = get_distances(distances, X, id)
    ids = [id]

    for i in range(1, size):
        id = torch.argmax(distances)
        distances = get_distances(distances, X, id)
        ids.append(id)

    X_cs.append(X[ids])
    y_cs.append(y[ids])
    X = torch.cat([X[:ids[0]], X[ids[0]+1:]], dim=0)
    y = torch.cat([y[:ids[0]], y[ids[0]+1:]], dim=0)

    return X_cs, y_cs, X, y

def get_distances(distances, X, id):
    for i in range(X.size(0)):
        current_dist = torch.norm(X[i, :] - X[id, :])
        distances[i] = torch.minimum(current_dist, distances[i])
    return distances