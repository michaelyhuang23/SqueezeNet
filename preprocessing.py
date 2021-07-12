import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
import cv2

class Graying(object):
    def __init__(self):
        super(Graying)
    def __call__(self, image):
        if image.shape[0]==1:
            image = image.repeat((3,1,1))
        return image

class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return 12500*2

    def __getitem__(self, index):
        if index<12500:
            img = cv2.imread(f'{self.root_dir}/cats/cat.{index}.jpg')
        else:
            img = cv2.imread(f'{self.root_dir}/dogs/dog.{index-12500}.jpg')
        minDim = min(img.shape[0],img.shape[1])
        img = cv2.resize(img,(int(img.shape[1]*224/minDim),int(img.shape[0]*224/minDim)))
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(0 if index<12500 else 1).float()
        return (img,label)     

def load_data():
    trans = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(224)])
    total_dataset = CatsDogsDataset('data/dogs-vs-cats',transform=trans)
    train_size = int(len(total_dataset)*0.7)
    val_size = int(len(total_dataset)*0.1)
    test_size = len(total_dataset)-train_size-val_size
    train_dataset,val_dataset,test_dataset = random_split(total_dataset, [train_size,val_size,test_size])
    train_data = DataLoader(train_dataset,batch_size=512,shuffle=True)
    test_data = DataLoader(test_dataset,batch_size=512,shuffle=True)
    val_data = DataLoader(val_dataset,batch_size=512,shuffle=True)
    return train_data, val_data, test_data
