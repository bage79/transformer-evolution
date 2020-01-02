import torch

if __name__ == '__main__':
    m1 = torch.FloatTensor([[1, 2], [3, 4]])
    m2 = torch.FloatTensor([[1, 1], [2, 2]])
    print(m1.mul(m2))
    print(m1.mean(dim=0))
    print(m1.max(dim=0).values)
    print(m1.max(dim=0).indices)
    print(m1.argmax(dim=0))
