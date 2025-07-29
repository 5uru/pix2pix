from tinygrad import Tensor, nn
from model import Unet, Discriminator
from dataset import get_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Initialize models
unet = Unet()
discriminator = Discriminator()

# Initialize optimizers
optimizer_unet = nn.optim.Adam(nn.state.get_parameters(unet), lr=0.0002, b1=0.5, b2=0.999)
optimizer_discriminator = nn.optim.Adam(nn.state.get_parameters(discriminator), lr=0.0002, b1=0.5, b2=0.999)

# Load dataset
train_dataset, val_dataset = get_dataset(batch_size=16)
X_train, Y_train = train_dataset
X_val, Y_val = val_dataset

# Loss history tracking
train_d_losses = []
train_g_losses = []
val_d_losses = []
val_g_losses = []

# Training loop
for epoch in tqdm(range(1000)):
    # Training phase
    epoch_d_losses = []
    epoch_g_losses = []

    for x, y in zip(X_train, Y_train):
        Tensor.training = True
        # Train discriminator
        real_data = Tensor.ones(x.shape[0])
        y_pred = discriminator(x, y).mean(axis=[2,3]).squeeze(dim=1)
        loss_real = y_pred.binary_crossentropy(real_data)

        fake_y = unet(x)
        y_pred_fake = discriminator(x, fake_y).mean(axis=[2,3]).squeeze(dim=1)
        loss_fake = y_pred_fake.binary_crossentropy(Tensor.zeros(y_pred_fake.shape))
        loss_discriminator = (loss_real + loss_fake) / 2

        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Train generator
        fake_y = unet(x)  # Recompute with updated weights
        labels = Tensor.ones(y.shape[0])
        y_pred = discriminator(x, fake_y).mean(axis=[2,3]).squeeze(dim=1)
        loss_gan = y_pred.binary_crossentropy(labels)
        l1_loss = (fake_y - y).abs().mean()
        loss_gen = loss_gan + 100 * l1_loss

        optimizer_unet.zero_grad()
        loss_gen.backward()
        optimizer_unet.step()

        # Store batch losses
        epoch_d_losses.append(loss_discriminator.numpy())
        epoch_g_losses.append(loss_gen.numpy())

    # Calculate average training losses for this epoch
    train_d_loss = np.mean(epoch_d_losses)
    train_g_loss = np.mean(epoch_g_losses)
    train_d_losses.append(train_d_loss)
    train_g_losses.append(train_g_loss)

    # Validation phase
    Tensor.training = False
    epoch_val_d_losses = []
    epoch_val_g_losses = []

    for x, y in zip(X_val, Y_val):
        # Discriminator validation
        real_data = Tensor.ones(x.shape[0])
        y_pred = discriminator(x, y).mean(axis=[2,3]).squeeze(dim=1)
        loss_real = y_pred.binary_crossentropy(real_data)

        fake_y = unet(x)
        y_pred_fake = discriminator(x, fake_y).mean(axis=[2,3]).squeeze(dim=1)
        loss_fake = y_pred_fake.binary_crossentropy(Tensor.zeros(y_pred_fake.shape))
        val_loss_discriminator = (loss_real + loss_fake) / 2

        # Generator validation
        y_pred = discriminator(x, fake_y).mean(axis=[2,3]).squeeze(dim=1)
        loss_gan = y_pred.binary_crossentropy(real_data)
        l1_loss = (fake_y - y).abs().mean()
        val_loss_gen = loss_gan + 100 * l1_loss

        epoch_val_d_losses.append(val_loss_discriminator.numpy())
        epoch_val_g_losses.append(val_loss_gen.numpy())

    # Calculate average validation losses
    val_d_loss = np.mean(epoch_val_d_losses)
    val_g_loss = np.mean(epoch_val_g_losses)
    val_d_losses.append(val_d_loss)
    val_g_losses.append(val_g_loss)

    # Print progress
    print(f"Epoch {epoch+1}/100")
    print(f"Train: D_loss={train_d_loss:.4f}, G_loss={train_g_loss:.4f}")
    print(f"Val: D_loss={val_d_loss:.4f}, G_loss={val_g_loss:.4f}")

# Plot losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_d_losses) + 1), train_d_losses, 'b-', label='Training')
plt.plot(range(1, len(val_d_losses) + 1), val_d_losses, 'r-', label='Validation')
plt.title('Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_g_losses) + 1), train_g_losses, 'b-', label='Training')
plt.plot(range(1, len(val_g_losses) + 1), val_g_losses, 'r-', label='Validation')
plt.title('Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()