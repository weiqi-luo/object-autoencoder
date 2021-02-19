# for data loading
import torch
import torchvision
import torchvision.transforms as transforms
from preprocess import ObjectPoseDataset, Rescale, RandomCrop, ToTensor
from torch.utils.data import DataLoader
# for net building 
from train import Net, imshow
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = './objectAutoEncoder_net.pth'


## TODO Load and normalize the test dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testdataset = ObjectPoseDataset(npy_file='../../data/interim/camera_poses.npy',
                                        root_dir='../../data/interim/',
                                        transform=transform)

testloader = DataLoader(testdataset, batch_size=4,shuffle=False, num_workers=4)


## TODO Load the network and trained parameters
net = Net()
net.load_state_dict(torch.load(PATH))
net.to(device)


## TODO Test the network on the test data
dataiter = iter(testloader)
data = dataiter.next()
images = data['image'].to(device)

# print test images
imshow(torchvision.utils.make_grid(images))

# prediction
outputs = net(images)
imshow(outputs)

# testing whole set
with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images = data['image'].to(device)
        outputs = net(images)

print('Accuracy of the network on the 10000 test images: ' % ()