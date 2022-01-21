import torch
import torch.nn as nn

# input image   64 x 64 x 3
# Conv1         32 x 32 x 16 (16=conv1's kernel number)
# Conv2         16 x 16 x 32 (32=conv2's kernel number)
# Conv3         8 x 8 x 64   (64=conv3's kernel number) 

class Auto_Encoder_Depth_v1(nn.Module):
    def __init__(self):
        super(Auto_Encoder_Depth_v1, self).__init__()

        self.dropout_layer = nn.Dropout2d(0.5)

        self.encoder = nn.Sequential(         
            # layer 1
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.decoder = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # layer 2
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # layer 3
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),  
            nn.BatchNorm2d(4),
            nn.Sigmoid()
        )        

    def forward(self, x):
        
        x = self.dropout_layer(x)
        latent = self.encoder(x)
        out = self.decoder(latent)
      
        return out
                

class Auto_Encoder_Depth_layer4(nn.Module):
    def __init__(self):
        super(Auto_Encoder_Depth_layer4, self).__init__()

        self.dropout_layer = nn.Dropout2d(0.1)

        self.encoder = nn.Sequential(         
            # layer 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.decoder = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # layer 2
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # original은 kernel_size=3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # layer 3
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # layer 4
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  
            nn.BatchNorm2d(3),
            nn.Sigmoid()            
        )        

    def forward(self, x):

        x = self.dropout_layer(x)
        latent = self.encoder(x)
        out = self.decoder(latent)
      
        return out
