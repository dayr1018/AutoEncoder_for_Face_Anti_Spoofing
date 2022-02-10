import torch.nn as nn

        # input image   64 x 64 x 3
# Conv1         32 x 32 x 16 (16=conv1's kernel number)
# Conv2         16 x 16 x 32 (32=conv2's kernel number)
# Conv3         8 x 8 x 64   (64=conv3's kernel number)

# dropout_rate = 0.25

class AutoEncoder_Original(nn.Module):
    def __init__(self):
        super(AutoEncoder_Original, self).__init__()
        
        self.encoder_layer = nn.Sequential(         
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.decoder_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # original은 kernel_size=3
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        latent = self.encoder_layer(x)
        out = self.decoder_layer(latent)
      
        return out
                
class AutoEncoder_Dropout(nn.Module):
    def __init__(self, use_drop = True, dropout_rate = 0.5):
        super(AutoEncoder_Dropout, self).__init__()

        self.dropout_layer = nn.Dropout2d(dropout_rate)
        self.use_drop = use_drop
 
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
            nn.MaxPool2d(2,2)
        )

        self.decoder = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # original은 kernel_size=3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # layer 2
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # layer 3
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )        

    def forward(self, x):

        if self.use_drop == True:
            x = self.dropout_layer(x)
        latent = self.encoder(x)
        out = self.decoder(latent)
      
        return out
                
class AutoEncoder_layer4(nn.Module):
    def __init__(self, use_drop = True, dropout_rate = 0.5):
        super(AutoEncoder_layer4, self).__init__()

        self.dropout_layer = nn.Dropout2d(dropout_rate)
        self.use_drop = use_drop

        self.encoder = nn.Sequential(         
            # layer 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            ## nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ## nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ## nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # layer 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ## nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.decoder = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            ## nn.BatchNorm2d(64),
            nn.ReLU(),
            # layer 2
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # original은 kernel_size=3
            ## nn.BatchNorm2d(32),
            nn.ReLU(),
            # layer 3
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            ## nn.BatchNorm2d(16),
            nn.ReLU(),
            # layer 4
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  
            ## nn.BatchNorm2d(3),
            nn.Sigmoid()            
        )        

    def forward(self, x):

        if self.use_drop == True:
            x = self.dropout_layer(x)
        latent = self.encoder(x)
        out = self.decoder(latent)
      
        return out
