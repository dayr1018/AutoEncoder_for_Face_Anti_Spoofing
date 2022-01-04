from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
from PIL import Image


class Face_Data(Dataset):
        def __init__(self, metadata_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_code/metadata/', 
                        data_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/data/' , datatxt = '', transform=None):
            self.metadata_root = metadata_root
            self.data_root = data_root
            self.transform = transform
            self.rgb_paths = []
            self.labels = []

            lines_in_txt = open(os.path.join(metadata_root, datatxt),'r')

            for line in lines_in_txt:
                line = line.rstrip() 
                split_str = line.split()

                rgb_path = os.path.join(data_root, split_str[0])
                label = split_str[1] 
                self.rgb_paths.append(rgb_path)
                self.labels.append(label)

        def __getitem__(self,index):
            rgb_path = self.rgb_paths[index]
            rgb_img = Image.open(rgb_path).convert('RGB')

            if self.transform is not None:
                rgb_img = self.transform(rgb_img)

            label = torch.as_tensor(int(self.labels[index]))
            
            return rgb_img, label, rgb_path

        def __len__(self):
            return len(self.rgb_paths)

def Facedata_Loader(train_size=64, test_size=64): 
    data_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_data=Face_Data(datatxt='MakeTextFileCode_RGB/train_data_list.txt', transform=data_transform)
    valid_data=Face_Data(datatxt='MakeTextFileCode_RGB/valid_data_list.txt', transform=data_transform)
    test_data=Face_Data(datatxt='MakeTextFileCode_RGB/test_data_list.txt', transform=data_transform) # test 데이터 세 종류 있음. 

    train_loader = DataLoader(dataset=train_data, batch_size=train_size, shuffle=True, num_workers=32)
    valid_loader = DataLoader(dataset=valid_data, batch_size=train_size, shuffle=True, num_workers=32)
    test_loader = DataLoader(dataset=test_data, batch_size=test_size, shuffle=True, num_workers=32)

    return train_loader, valid_loader, test_loader

