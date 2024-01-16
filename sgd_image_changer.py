#Change an image into another image using sgd

import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

plt.figure()
f, axarr = plt.subplots(10,1) 

device = "cuda" if torch.cuda.is_available() else "cpu"
image2 = transforms.ToTensor()(Image.open("dog.png")).unsqueeze(0).to(device)
image2 = transforms.Resize(224)(image2)
image = torch.zeros(3,224,224, requires_grad=True)

print(image.shape, image2.shape)
optimizer = optim.SGD([image], lr=1, momentum=0.9)
criterion = nn.CrossEntropyLoss()
for epoch in range(120):
    optimizer.zero_grad()
    loss = torch.linalg.vector_norm(image - image2)
    print(epoch, loss)

    loss.backward(retain_graph=True)
    optimizer.step()
    if epoch % 10 == 0:
        axarr[epoch//10].imshow(image.detach()[:,:,:].permute(1, 2, 0))

print('Finished Training')


