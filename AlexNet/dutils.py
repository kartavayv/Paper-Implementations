from torchvision import datasets, transforms
import torch as t
from torch.utils.data import Dataset, DataLoader


normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

tf_train = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), normalize,
])

tf_val = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(227),
    transforms.ToTensor(), normalize,
])


train = datasets.ImageFolder("Dataset/imagenet-mini/train", tf_train)
val   = datasets.ImageFolder("Dataset/imagenet-mini/val", tf_val)


train_loader = DataLoader(train, shuffle=True, num_workers=4, batch_size=128, pin_memory=True)
val_loader   = DataLoader(val, shuffle=False, num_workers=4, batch_size=128)