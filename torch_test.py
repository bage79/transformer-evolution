import torch

if __name__ == '__main__':
    m1 = torch.FloatTensor([[1, 2], [3, 4]])
    m2 = torch.FloatTensor([[1], [2]])
    m22 = torch.FloatTensor([[1, 1], [2, 2]])
    print(m1.mul(m2))
    print(m1.mul(m22))
    print()
    print(m1)
    print('mean', m1.mean(dim=1))
    print('max[0]', m1.max(dim=1).values)
    print('max[1]', m1.max(dim=1).indices)
    print(m1.argmax(dim=0))
