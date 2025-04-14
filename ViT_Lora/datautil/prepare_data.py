import torch
from torch.utils.data import Dataset, ConcatDataset
from datautil.datasplit import getdataloader
from medmnist import OrganAMNIST,OrganCMNIST,OrganSMNIST
from PIL import Image
import torchvision.transforms as transforms


def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'medmnistA': 'medmnistA', 'medmnistC': 'medmnistC','medmnistS': 'medmnistS'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def getlabeldataloader(args, data):
    trl, val, tel = getdataloader(args, data)
    trd, vad, ted = [], [], []
    for i in range(len(trl)):
        trd.append(torch.utils.data.DataLoader(
            trl[i], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))
    return trd, vad, ted

class MedMnistDataset(Dataset):
    def __init__(self,args,datasets):
        self.data = []
        self.targets = []
        self.Trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for dataset in datasets:
            self.data.extend(dataset.imgs)
            self.targets.extend(dataset.labels.flatten())

        subset_size = int(len(self.data) * args.datapercent) 
        indices = args.random_state.choice(len(self.data), subset_size, replace=False)  

        self.data = [self.data[idx] for idx in indices]
        self.targets = [self.targets[idx] for idx in indices]

    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx]).convert('RGB')
        img = self.Trans(img)
        return img, self.targets[idx]


def medmnistA(args):
    data_train = OrganAMNIST(split="train", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data_test = OrganAMNIST(split="test", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data_val = OrganAMNIST(split="val", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data = MedMnistDataset(args,[data_train, data_test,data_val])
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted
def medmnistC(args):
    data_train = OrganCMNIST(split="train", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data_test = OrganCMNIST(split="test", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data_val = OrganCMNIST(split="val", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data = MedMnistDataset(args,[data_train, data_test,data_val])
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted
def medmnistS(args):
    data_train = OrganSMNIST(split="train", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data_test = OrganSMNIST(split="test", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data_val = OrganSMNIST(split="val", root="/home/cxy/wph/data/MedMNIST/", size=224)
    data = MedMnistDataset(args,[data_train, data_test,data_val])
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted







