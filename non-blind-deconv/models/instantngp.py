import torch
import tinycudann as tcnn
import torch.nn as nn
import torch.nn.functional as F

class InstantNGPModel(torch.nn.Module):
    """
    Complete Instant-NGP model using tiny-cuda-nn with separate encoder and decoder
    Supports both 2D and 3D inputs
    """
    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        learn_variance=True,
        n_input_dims=3,
    ):
        super().__init__()
        
        # Default encoder config for tiny-cuda-nn HashEncoding
        if encoder_config is None:
            encoder_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 20,
                "base_resolution": 16,
                "per_level_scale": 1.4142135624,
            }
        
        # Default decoder config for tiny-cuda-nn Network
        if decoder_config is None:
            decoder_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2,
            }
        
        self.encoder_config = encoder_config
        self.n_input_dims = n_input_dims

        # Progressive training: gradually unlock high-frequency levels
        self.n_levels = encoder_config["n_levels"]
        self.n_features_per_level = encoder_config["n_features_per_level"]
        self.current_max_level = self.n_levels  # Will be updated during training

        # Create encoder using tiny-cuda-nn
        self.encoder = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config=encoder_config,
        )
        
        # Get encoder output dimension
        encoder_output_dim = self.encoder.n_output_dims
        
        # Create decoder using tiny-cuda-nn
        decoder_config["n_input_dims"] = encoder_output_dim
        if learn_variance:
            decoder_config["n_output_dims"] = 2
        else:
            decoder_config["n_output_dims"] = 1
        self.learn_variance = learn_variance
        
        self.decoder = tcnn.Network(
            n_input_dims=encoder_output_dim,
            n_output_dims=decoder_config["n_output_dims"],
            network_config=decoder_config,
        )
        
        print(f"InstantNGPModel initialized with tiny-cuda-nn")
        print(f"  Encoder output dim: {encoder_output_dim}")
        print(f"  Encoder config: {encoder_config}")
        print(f"  Decoder config: {decoder_config}")
        print(f"  Progressive training enabled: masking high-frequency levels initially")
    
    def set_max_level(self, max_level):
        """Set the maximum level to use (for progressive training)"""
        self.current_max_level = min(max_level, self.n_levels)

    def forward(self, x, variance=None, stochastic_alpha=None): #variance=1.6169893497):
        """
        x: [N, 2] or [N, 3] coordinates in [0, 1]
        returns: [N] density values
        
        Note: tiny-cuda-nn expects inputs in [0, 1]
        """
        coords = x
        
        # --- Stochastic Preconditioning (Strictly following your snippet logic) ---
        if stochastic_alpha is not None and stochastic_alpha > 0:
            # Generate Gaussian noise
            noise = stochastic_alpha * torch.normal(torch.zeros_like(coords), torch.ones_like(coords))
            
            coords = coords + noise
            
            # reflect around boundary as described in paper Section 4.1
            coords = coords % 2
            
            # In-place modification for reflection
            mask = coords > 1
            coords[mask] = 2 - coords[mask]

        with torch.amp.autocast('cuda'):
            # Encode (x is already in [0, 1])
            encoded = self.encoder(coords)
        
            # Progressive training: mask high-frequency levels
            # Reshape to [N, n_levels, n_features_per_level]
            encoded = encoded.view(encoded.shape[0], self.n_levels, self.n_features_per_level)
            
            # Mask levels beyond current_max_level
            if self.current_max_level < self.n_levels:
                mask = torch.ones_like(encoded)
                mask[:, self.current_max_level:, :] = 0.0
                encoded = encoded * mask

            # Add variance to encoded if provided
            if variance is not None:
                levels = self.encoder_config["per_level_scale"] ** torch.arange(0, self.encoder_config["n_levels"], device=encoded.device, dtype=encoded.dtype).view(1, self.encoder_config["n_levels"], 1)
                # Compute the factors
                factor = 1.0 / (torch.sqrt(torch.tensor([0.1], device=encoded.device)) * variance * levels)  # (N, n_levels, 1)
                # Use erf for weighting
                weights = torch.erf(factor)  # (N, n_levels, 1)
                # Multiply weights by encoded
                encoded = encoded * weights
            
            # Reshape back to original shape
            encoded = encoded.view(encoded.shape[0], -1)
            
            # Decode
            output = self.decoder(encoded)

            # Convert to float32 for stability
            output = output.float()
            
            # Apply Softplus to ensure output > 0 (strictly positive)
            output = F.softplus(output)

            if self.learn_variance:
                return output[..., 0], output[..., 1]
            
        return output.squeeze(-1), None
    

class InstantNGPTorchModel(nn.Module):
    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        learn_variance=True,
        n_input_dims=3,
    ):
        super().__init__()
        
        # Default encoder config for tiny-cuda-nn HashEncoding
        if encoder_config is None:
            encoder_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 20,
                "base_resolution": 16,
                "per_level_scale": 1.4142135624,
            }
        
        # Default decoder config for tiny-cuda-nn Network
        if decoder_config is None:
            decoder_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2,
            }
        
        self.encoder_config = encoder_config
        self.n_input_dims = n_input_dims
        
        # Progressive training: gradually unlock high-frequency levels
        self.n_levels = encoder_config["n_levels"]
        self.n_features_per_level = encoder_config["n_features_per_level"]
        self.current_max_level = self.n_levels  # Will be updated during training

        # Create encoder using tiny-cuda-nn
        self.encoder = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config=encoder_config,
        )
        
        # Get encoder output dimension
        encoder_output_dim = self.encoder.n_output_dims
        
        # Create decoder using tiny-cuda-nn
        decoder_config["n_input_dims"] = encoder_output_dim
        if learn_variance:
            decoder_config["n_output_dims"] = 2
        else:
            decoder_config["n_output_dims"] = 1
        self.learn_variance = learn_variance
        
        hidden_dim = decoder_config.get("n_neurons", 128)
        num_layers = decoder_config.get("n_hidden_layers", 2)
        out_dim = 2 if self.learn_variance else 1

        layers = []
        in_dim = encoder_output_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_dim))

        self.decoder = nn.Sequential(*layers)

        
        print(f"InstantNGPTorchModel initialized with tiny-cuda-nn and PyTorch MLP")
        print(f"  Encoder output dim: {encoder_output_dim}")
        print(f"  Encoder config: {encoder_config}")
        print(f"  Decoder config: {decoder_config}")
        print(f"  Progressive training enabled: masking high-frequency levels initially")
    
    def set_max_level(self, max_level):
        """Set the maximum level to use (for progressive training)"""
        self.current_max_level = min(max_level, self.n_levels)
    
    def forward(self, x, variance=None, stochastic_alpha=None):
        coords = x
        
        # --- Stochastic Preconditioning (Strictly following your snippet logic) ---
        if stochastic_alpha is not None and stochastic_alpha > 0:
            # Generate Gaussian noise
            noise = stochastic_alpha * torch.normal(torch.zeros_like(coords), torch.ones_like(coords))
            
            coords = coords + noise
            
            # reflect around boundary as described in paper Section 4.1
            coords = coords % 2
            
            # In-place modification for reflection
            mask = coords > 1
            coords[mask] = 2 - coords[mask]

        # Encode (x is already in [0, 1])
        with torch.amp.autocast('cuda'):
            encoded = self.encoder(coords)
            
            # Progressive training: mask high-frequency levels
            # Reshape to [N, n_levels, n_features_per_level]
            encoded = encoded.view(encoded.shape[0], self.n_levels, self.n_features_per_level)
            
            # Mask levels beyond current_max_level
            if self.current_max_level < self.n_levels:
                mask = torch.ones_like(encoded)
                mask[:, self.current_max_level:, :] = 0.0
                encoded = encoded * mask

            # Add variance to encoded if provided
            if variance is not None:
                levels = self.encoder_config["per_level_scale"] ** torch.arange(0, self.encoder_config["n_levels"], device=encoded.device, dtype=encoded.dtype).view(1, self.encoder_config["n_levels"], 1)
                # Compute the factors
                factor = 1.0 / (torch.sqrt(torch.tensor([0.1], device=encoded.device)) * variance * levels)  # (N, n_levels, 1)
                # Use erf for weighting
                weights = torch.erf(factor)  # (N, n_levels, 1)
                # Multiply weights by encoded
                encoded = encoded * weights
            
            # Reshape back to original shape
            encoded = encoded.view(encoded.shape[0], -1)
            
            # Decode
            output = self.decoder(encoded)

            # Convert to float32 for stability
            output = output.float()
            
            # Apply Softplus to ensure output > 0 (strictly positive)
            #output = F.softplus(output)

            if self.learn_variance:
                return output[..., 0], output[..., 1]
        
        return output.squeeze(-1), None