import torch

data = torch.load("/data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/debug.pth")

indices_list = data['dynamic']
means_list = data['means']


length = len(indices_list)

for i in range(length):
    indices = indices_list[i]
    means = means_list[i]
    means = means[:,indices,:]
    # print(means.shape)
    print(means.device)
    