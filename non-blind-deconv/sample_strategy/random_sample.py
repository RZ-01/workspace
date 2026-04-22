import torch


def sample_random_coords(num_pixels, data_shape, device='cpu'):
    # sample only this region crop_region 
    # x (0, 512)
    # y (0, 512)
    # z (600, 800)
    if len(data_shape) == 3:
        z = torch.randint(950, 1150, (num_pixels,), device=device)
        y = torch.randint(0, 1300, (num_pixels,), device=device)
        x = torch.randint(0, 1300, (num_pixels,), device=device)
        #z = torch.randint(0, data_shape[0], (num_pixels,), device=device)
        #y = torch.randint(0, data_shape[1], (num_pixels,), device=device)
        #x = torch.randint(0, data_shape[2], (num_pixels,), device=device)
        #z = torch.randint(600, 800, (num_pixels,), device=device)
        #y = torch.randint(0, 512, (num_pixels,), device=device)
        #x = torch.randint(0, 512, (num_pixels,), device=device)
        coords = torch.stack([z, y, x], dim=1)
    else:
        # sample only this region crop_region = (13000, 13000 + 4096, 32500, 32500 + 4096)
        y = torch.randint(13000, 13000 + 4096, (num_pixels,), device=device)
        x = torch.randint(32500, 32500 + 4096, (num_pixels,), device=device)
        coords = torch.stack([y, x], dim=1) 
    return coords.long()
    