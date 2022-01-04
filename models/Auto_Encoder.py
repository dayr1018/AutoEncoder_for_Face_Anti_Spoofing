import torch.nn as nn
    
class Auto_Encoder(nn.Module):
    def __init__(self):
        super(Auto_Encoder, self).__init__()
        
        # input image   64 x 64 x 3
        # Conv1         32 x 32 x 16 (16=conv1's kernel number)
        # Conv2         16 x 16 x 32 (32=conv2's kernel number)
        # Conv3         8 x 8 x 64   (64=conv3's kernel number)

        # encoder
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(16),
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

        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        latent = self.encoder_layer3(x)
        x = self.decoder_layer1(latent)
        x = self.decoder_layer2(x)
        out = self.decoder_layer3(x)

        return out
