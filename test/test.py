#  라이브러리
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from PIL import Image

class Face_Data(Dataset):
        def __init__(self, metadata_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/anomaly-detection_code/metadata/', 
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
            
            return rgb_img, label

        def __len__(self):
            return len(self.rgb_paths)

def Facedata_Loader(train_size=64, test_size=64): 
    data_transform = transforms.Compose([
        transforms.Resize((124,124)),
        # transforms.CenterCrop((112,112)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_data=Face_Data(datatxt='MakeTextFileCode_RGB/train_data_list.txt', transform=data_transform)
    test_data=Face_Data(datatxt='MakeTextFileCode_RGB/test_data_list.txt', transform=data_transform) # test 데이터 세 종류 있음. 

    train_loader = DataLoader(dataset=train_data, batch_size=train_size, shuffle=True, num_workers=32)
    test_loader = DataLoader(dataset=test_data, batch_size=test_size, shuffle=True, num_workers=32)

    return train_loader, test_loader
 
# 미리 만들어둔 모델 불러오기
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
 
    def forward(self, x):
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        return x
 
 
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
 
        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU()
        )        

        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )  

        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            # nn.ReLU()
            nn.Sigmoid()
        )  
 
    def forward(self, x):
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = self.decoder_layer3(x)
        return x

 
#  이미지를 저장할 폴더 생성
if not os.path.exists('./AE_img'):
    os.mkdir('./AE_img')
 
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
 
 
img_transform = transforms.Compose([
    transforms.ToTensor()
])
 
#  Hyper Parameter 설정
num_epochs = 10
batch_size = 128
learning_rate = 1e-3
 
 
#  맨 처음 한번만 다운로드 하기
# dataset = MNIST('./data', transform=img_transform, download=True)
 
#  데이터 불러오기
# dataset = MNIST('./data', transform=img_transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)
train_loader, test_loader = Facedata_Loader(train_size=64, test_size=64)
 
#  모델 설정
encoder = encoder().cuda().train()
decoder = decoder().cuda().train()
 
 
#  모델 Optimizer 설정
criterion = nn.MSELoss()
encoder_optimizer = torch.optim.Adam( encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam( decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
 
 
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data  # label 은 가져오지 않는다.
        # img = img.view(img.size(0), -1)
        # img = Variable(img).cuda()

        img.cuda()


        # image = rgb[batch].numpy()
        # # plt.imshow(np.transpose((image*255).astype(np.uint8), (1,2,0)))
        # plt.imshow(np.transpose(image.astype('float32'), (1,2,0)))
        # plt.show()
        # # print(rgb)
        # output = model(rgb)
        
        # imageout = output[0].cpu().detach().numpy()
        # plt.imshow(np.transpose(imageout.astype('float32')/255, (1,2,0)))
        # plt.show()



        # ===================forward=====================
        latent_z = encoder(img)
        output = decoder(latent_z )
        # ===================backward====================
        loss = criterion(output, img)
 
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, float(loss.data) ))
 
    if epoch % 10 == 0:
        # pic = to_img(output.cpu().data)
        pic = output.cpu().data
        pic = pic.view(pic.size(0), 1, 28, 28)
 
        save_image(pic, './AE_img/output_image_{}.png'.format(epoch))
 
#  모델 저장
torch.save(encoder.state_dict(), './encoder.pth')
torch.save(decoder.state_dict(), './decoder.pth')