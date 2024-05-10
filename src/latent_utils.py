import torch
import torch.nn as nn
import torch.nn.functional as F
# keep latents with n_keep
def get_latent(img, nc_model, device, n_keep=12):
        ps_layer = nn.PixelShuffle(2)
        img = img.to(device)
        # print(img.shape)
        stats_all = nc_model.forward_get_latents(img)
        latents = [stats_all[latent_block_index]['z'] for latent_block_index in range(12)]
        if n_keep == 12:
                output = torch.cat((F.interpolate(latents[0], 4),latents[1], latents[2]), 1)
                output = torch.cat((F.interpolate(output, 8),latents[3], latents[4],latents[5], latents[6]), 1)
                output = torch.cat((F.interpolate(output, 16),latents[7], latents[8],latents[9], latents[10], latents[11]), 1)
                output = ps_layer(output)
        elif n_keep == 1:
                output = latents[0]
        return output