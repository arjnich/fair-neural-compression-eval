import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 

# keep latents with n_keep
def get_latent(img, nc_model, device, n_keep=12):
        ps_layer = nn.PixelShuffle(2)
        img = img.to(device)
        # print(img.shape)
        stats_all = nc_model.forward_get_latents(img)
        latents = [stats_all[latent_block_index]['z'] for latent_block_index in range(12)]
        if n_keep == 12:
                # output dimension is [19, 32, 32] after PS.
                output = torch.cat((F.interpolate(latents[0], 4),latents[1], latents[2]), 1)
                output = torch.cat((F.interpolate(output, 8),latents[3], latents[4],latents[5], latents[6]), 1)
                output = torch.cat((F.interpolate(output, 16),latents[7], latents[8],latents[9], latents[10], latents[11]), 1)
                output = ps_layer(output)
        elif n_keep == 9:
                # output dimension is [16, 32, 32] after PS.
                output = torch.cat((F.interpolate(latents[0], 4),latents[1], latents[2]), 1)
                output = torch.cat((F.interpolate(output, 8),latents[3], latents[4],latents[5], latents[6]), 1)
                output = torch.cat((F.interpolate(output, 16),latents[7], latents[8]), 1)
                output = ps_layer(output)
        elif n_keep == 6:
                # output dimension is [50, 8, 8]
                output = torch.cat((F.interpolate(latents[0], 4),latents[1], latents[2]), 1)
                output = torch.cat((F.interpolate(output, 8),latents[3], latents[4],latents[5]), 1)
        elif n_keep == 3:
                # output dimension is [32, 4, 4]
                output = torch.cat((F.interpolate(latents[0], 4),latents[1], latents[2]), 1)
        elif n_keep == 1:
                # output dimensino is [16, 1, 1]
                output = latents[0]
        return output

def train_numerical_rfw_latents(
        nc_model,
        model, 
        num_epochs, 
        lr, 
        train_loader, 
        valid_loader,
        device, 
        n_keep, 
        writer,
        patience=5  # Number of epochs to wait for improvement in validation loss before stopping
    ):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    valid_losses = []

    best_valid_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, targets, races in train_loader:
                # print(inputs.shape)
                latents = get_latent(inputs, nc_model, device, n_keep=n_keep)
                # inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(latents)
                loss = 0
                for i, head in enumerate(outputs):
                    loss += criterion(outputs[head], targets[:, i].to(torch.int64))
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
                avg_train_loss = running_train_loss / ((pbar.n + 1) * len(latents))  # Compute average loss
                pbar.set_postfix(loss=avg_train_loss)
                pbar.update(1)
        print(f'Epoch {i + 1} train loss : {avg_train_loss}')
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', loss, epoch)
        # Validation phase
        model.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
                for inputs, targets, races in valid_loader:
                    latents = get_latent(inputs, nc_model, device, n_keep=n_keep)
                    # inputs, targets = inputs.to(device).float(), targets.to(device)
                    targets = targets.to(device)
                    outputs = model(latents)
                    loss = 0
                    for i, head in enumerate(outputs):
                        loss += criterion(outputs[head], targets[:, i].to(torch.int64))
                    running_valid_loss += loss.item() * inputs.size(0)
                    avg_valid_loss = running_valid_loss / ((pbar.n + 1) * len(inputs))  # Compute average validation loss
                    pbar.set_postfix(valid_loss=avg_valid_loss)
                    pbar.update(1)
        print(f'Epoch {epoch + 1} valid loss : {avg_valid_loss}')
        valid_losses.append(avg_valid_loss)
        writer.add_scalar('Loss/val', avg_valid_loss, epoch)
        
        # Check for early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, epoch+1, train_losses, valid_losses
