from tinygrad import Tensor, nn
import numpy as np
from tinygrad.helpers import colored

class Encoder:
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(4,4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=2, padding=1)
        self.norm1 = nn.InstanceNorm(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4,4), stride=2, padding=1)
        self.norm2 = nn.InstanceNorm(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(4,4), stride=2, padding=1)
        self.norm3 = nn.InstanceNorm(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(4,4), stride=2, padding=1)
        self.norm4 = nn.InstanceNorm(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(4,4), stride=2, padding=1)
        self.norm5 = nn.InstanceNorm(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(4,4), stride=1, padding=1)
        self.norm6 = nn.InstanceNorm(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=(4,4), stride=1, padding=1)
        self.norm7 = nn.InstanceNorm(512)

    def __call__(self, x: Tensor):
        x = self.conv1(x).leaky_relu()
        skip_connection = [x]
        x = self.conv2(x)
        x = self.norm1(x).leaky_relu()
        skip_connection.append(x)
        x = self.conv3(x)
        x = self.norm2(x).leaky_relu()
        skip_connection.append(x)
        x = self.conv4(x)
        x = self.norm3(x).leaky_relu()
        skip_connection.append(x)
        x = self.conv5(x)
        x = self.norm4(x).leaky_relu()
        skip_connection.append(x)
        x = self.conv6(x)
        x = self.norm5(x).leaky_relu()
        skip_connection.append(x)
        x = self.conv7(x)
        x = self.norm6(x).leaky_relu()
        skip_connection.append(x)
        x = self.conv8(x)
        x = self.norm7(x).leaky_relu()
        skip_connection.append(x)
        return x, skip_connection


class Decoder:
    def __init__(self):
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(4,4), stride=1, padding=1)
        self.norm1 = nn.InstanceNorm(512*2)
        self.deconv2 = nn.ConvTranspose2d(512*2, 512, kernel_size=(4,4), stride=1, padding=1)
        self.norm2 = nn.InstanceNorm(512*2)
        self.deconv3 = nn.ConvTranspose2d(512*2, 512, kernel_size=(4,4), stride=2, padding=1)
        self.norm3 = nn.InstanceNorm(512*2)
        self.deconv4 = nn.ConvTranspose2d(512*2, 512, kernel_size=(4,4), stride=2, padding=1)
        self.norm4 = nn.InstanceNorm(512*2)
        self.deconv5 = nn.ConvTranspose2d(512*2, 256, kernel_size=(4,4), stride=2, padding=1)
        self.norm5 = nn.InstanceNorm(256*2)  # Changed from norm4 to norm5
        self.deconv6 = nn.ConvTranspose2d(256*2, 128, kernel_size=(4,4), stride=2, padding=1)
        self.norm6 = nn.InstanceNorm(128*2)  # Changed from norm5 to norm6
        self.deconv7 = nn.ConvTranspose2d(128*2, 64, kernel_size=(4,4), stride=2, padding=1)
        self.norm7 = nn.InstanceNorm(64*2)   # Changed from norm6 to norm7
        self.deconv8 = nn.ConvTranspose2d(64*2, 3, kernel_size=(4,4), stride=2, padding=1)


    def __call__(self, x: Tensor, skip_connections: list):
        skip_connection = skip_connections[-2::-1]
        x = self.deconv1(x)
        x = x.cat(skip_connection[0], dim=1)
        x = self.norm1(x).dropout(0.5).relu()
        x = self.deconv2(x)
        x = x.cat(skip_connection[1], dim=1)
        x = self.norm2(x).dropout(0.5).relu()
        x = self.deconv3(x)
        x = x.cat(skip_connection[2], dim=1)
        x = self.norm3(x).dropout(0.5).relu()
        x = self.deconv4(x)
        x = x.cat(skip_connection[3], dim=1)
        x = self.norm4(x).relu()
        x = self.deconv5(x)
        x = x.cat(skip_connection[4], dim=1)
        x = self.norm5(x).relu()  # Changed from norm4 to norm5
        x = self.deconv6(x)
        x = x.cat(skip_connection[5], dim=1)
        x = self.norm6(x).relu()  # Changed from norm5 to norm6
        x = self.deconv7(x)
        x = x.cat(skip_connection[6], dim=1)
        x = self.norm7(x).relu()  # Changed from norm6 to norm7
        x = self.deconv8(x)
        x = x.tanh()

        return x

class Unet:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x: Tensor):
        out, skip = self.encoder(x)
        return self.decoder(out, skip)


class Discriminator:
    def __init__(self):
        self.conv1 = nn.Conv2d(3*2, 64, kernel_size=(4,4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=2, padding=1)
        self.norm1 = nn.InstanceNorm(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4,4), stride=2, padding=1)
        self.norm2 = nn.InstanceNorm(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(4,4), stride=1, padding=1)
        self.norm3 = nn.InstanceNorm(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=(4,4), stride=1, padding=1)


    def __call__(self, x: Tensor, y: Tensor):
        x = x.cat(y, dim=1)
        x = self.conv1(x).leaky_relu()
        x = self.conv2(x)
        x = self.norm1(x).leaky_relu()
        x = self.conv3(x)
        x = self.norm2(x).leaky_relu()
        x = self.conv4(x)
        x = self.norm3(x).leaky_relu()
        x = self.conv5(x)
        return x.sigmoid()

if __name__ == "__main__":
    encoder = Encoder()
    x = Tensor(np.random.randn(1, 3, 256, 256))
    out, skip = encoder(x)
    print(colored("Encoder output shape:", "green"), out.shape)
    print(colored("Skip connections shapes:", "blue"), [s.shape for s in skip])

    decoder = Decoder()
    out_dec = decoder(out, skip)
    print(colored("Decoder output shape:", "green"), out_dec.shape)

    unet = Unet()
    out_unet = unet(x)
    print(colored("UNet output shape:", "green"), out_unet.shape)

    discriminator = Discriminator()
    out_disc = discriminator(x, out_unet)
    print(colored("Discriminator output shape:", "green"), out_disc.shape)