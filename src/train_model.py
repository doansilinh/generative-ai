import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import d_model
import data_loader
import g_model
import setting

# Count total data
print(setting.device)
num_test = int(len(os.listdir(setting.raw_path)) * setting.test_ratio)
num_train = int((int(len(os.listdir(setting.raw_path))) - num_test))

print("Number of train images:", num_train)
print("Number of test images:", num_test)

# Train , test split
random.seed(setting.rand_seed)
train_indexs = np.array(random.sample(range(num_test + num_train), num_train))
mask = np.ones(num_train + num_test, dtype=bool)
mask[train_indexs] = False
inputs = [
    "IMG_{}.jpg".format(i) for i in range(1, len(os.listdir(setting.input_path)) + 1)
]
raws = ["IMG_{}.jpg".format(i) for i in range(1, len(os.listdir(setting.raw_path)) + 1)]
train_input_img_paths = np.array(inputs)[train_indexs]
train_raw_img_path = np.array(raws)[train_indexs]
test_input_img_paths = np.array(inputs)[mask]
test_raw_img_path = np.array(raws)[mask]


print("Ready to load")
# Set train, test loader
train_loader = data_loader.dataset(
    batch_size=setting.batch_size,
    img_size=setting.img_size,
    input_image_paths=train_input_img_paths,
    raw_image_paths=train_raw_img_path,
    input_path=setting.input_path,
    raw_path=setting.raw_path,
)
test_loader = data_loader.dataset(
    batch_size=setting.batch_size,
    img_size=setting.img_size,
    input_image_paths=test_input_img_paths,
    raw_image_paths=test_raw_img_path,
    input_path=setting.input_path,
    raw_path=setting.raw_path,
)


# Init model, optimizer and scheduler
discriminator = d_model.DModel().to(setting.device)
generator = g_model.GModel().to(setting.device)

bce = nn.BCEWithLogitsLoss()
l1loss = nn.L1Loss()

optimizer_D = optim.Adam(discriminator.parameters(), lr=setting.learning_rate_D)
optimizer_G = optim.Adam(generator.parameters(), lr=setting.learning_rate_G)

scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.1)
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.1)

best_generator_epoch_val_loss, best_discriminator_epoch_val_loss = np.inf, np.inf

os.makedirs("model", exist_ok=True)
os.makedirs("model", exist_ok=True)

print("Start training")
# Loop through N epochs
for epoch in range(setting.num_epochs):
    print("#" * 20, "Epoch = ", epoch + 1)
    # Swicth to train mode
    discriminator.train()
    generator.train()

    discriminator_epoch_loss, generator_epoch_loss = 0, 0

    # Loop through all data
    for inputs, raws in train_loader:
        print(".")
        inputs, true_images = inputs.to(setting.device), raws.to(setting.device)

        ################ Train D Model
        optimizer_D.zero_grad()

        #  Make fake image from G_model
        fake_images = generator(inputs)  # .detach()

        # Predict Fake/Real from D_model with fake image (hope will return 0) and count loss
        pred_fake = discriminator(fake_images).to(setting.device)
        loss_fake = bce(
            pred_fake, torch.zeros(setting.batch_size, device=setting.device)
        )

        # Predict Fake/Real from D_model with real image (hope will return 1) and count loss
        pred_real = discriminator(true_images).to(setting.device)
        loss_real = bce(
            pred_real, torch.ones(setting.batch_size, device=setting.device)
        )

        # Loss of D_model = avg 2 losses
        loss_D = (loss_fake + loss_real) / 2

        # backpropagation to caculate derivative
        loss_D.backward()
        optimizer_D.step()

        # Sum D_Model loss for this epoch
        discriminator_epoch_loss += loss_D.item()

        ################ Train G Model
        optimizer_G.zero_grad()

        # Gen fake image from inputs
        fake_images = generator(inputs)  # .detach()

        # Predict fake/real from D_model, caculate loss D_Model
        pred_fake = discriminator(fake_images).to(setting.device)
        loss_G_bce = bce(pred_fake, torch.ones_like(pred_fake, device=setting.device))

        # Caculate L1 loss, MAE between fake image and targets
        loss_G_l1 = l1loss(fake_images, true_images) * 100

        # Sum losses
        loss_G = loss_G_bce + loss_G_l1

        # Backpropagation to caculate derivative
        loss_G.backward()
        optimizer_G.step()

        # Sum GLoss
        generator_epoch_loss += loss_G.item()

    # Caculate D and G loss for this epoch
    discriminator_epoch_loss /= len(train_loader)
    generator_epoch_loss /= len(train_loader)

    print("#" * 20, "Start eval")
    # Switch to eval model
    discriminator.eval()
    generator.eval()

    discriminator_epoch_val_loss, generator_epoch_val_loss = 0, 0

    with torch.no_grad():
        for inputs, raws in test_loader:
            inputs, raws = inputs.to(setting.device), raws.to(setting.device)

            fake_images = generator(inputs).detach()
            pred_fake = discriminator(fake_images).to(setting.device)

            loss_G_bce = bce(
                pred_fake, torch.ones_like(pred_fake, device=setting.device)
            )
            loss_G_l1 = l1loss(fake_images, raws) * 100
            loss_G = loss_G_bce + loss_G_l1
            loss_D = bce(
                pred_fake.to(setting.device),
                torch.zeros(setting.batch_size, device=setting.device),
            )

            discriminator_epoch_val_loss += loss_D.item()
            generator_epoch_val_loss += loss_G.item()

    discriminator_epoch_val_loss /= len(test_loader)
    generator_epoch_val_loss /= len(test_loader)

    print(
        f"------Epoch [{epoch + 1}/{setting.num_epochs}]------\nTrain Loss D: {discriminator_epoch_loss:.4f}, Val Loss D: {discriminator_epoch_val_loss:.4f}"
    )
    print(
        f"Train Loss G: {generator_epoch_loss:.4f}, Val Loss G: {generator_epoch_val_loss:.4f}"
    )

    # Save best weight
    if discriminator_epoch_val_loss < best_discriminator_epoch_val_loss:
        # discriminator_epoch_val_loss = best_discriminator_epoch_val_loss
        best_discriminator_epoch_val_loss = discriminator_epoch_val_loss
        torch.save(discriminator.state_dict(), "model/discriminator.pth")
        print("Save D at epoch ", epoch + 1)
    if generator_epoch_val_loss < best_generator_epoch_val_loss:
        # generator_epoch_val_loss = best_generator_epoch_val_loss
        best_generator_epoch_val_loss = generator_epoch_val_loss
        torch.save(generator.state_dict(), "model/generator.pth")
        print("Save G at epoch ", epoch + 1)
