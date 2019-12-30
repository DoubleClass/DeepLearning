
import torch
import matplotlib.pyplot as plt
from sklearn import svm, neighbors
import torchvision
import torchvision.transforms as transforms


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   
    transforms.RandomHorizontalFlip(),      
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)
images = images.view(5000, -1)
images = list(images.numpy())
labels = list(labels.numpy())

dataiter2 = iter(testloader)
images_test, labels_test = dataiter2.next()
images_test = images_test.view(100, -1)
images_test = list(images_test.numpy())
labels_test = list(labels_test.numpy())


model = neighbors.KNeighborsClassifier(n_neighbors=15)
# model = svm.SVC()
model.fit(images, labels)
score = model.score(images_test, labels_test)
print(score)
