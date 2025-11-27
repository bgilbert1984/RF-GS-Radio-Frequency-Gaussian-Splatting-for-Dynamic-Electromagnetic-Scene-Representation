import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class NeuralCorrespondenceField(nn.Module):
    """
    Neural Correspondence Field (NCF) for tracking motion and changes in RF signals over time.
    Maps spatial-temporal coordinates to motion vectors and correspondence confidence.
    """
    
    def __init__(
        self,
        encoding_dim: int = 8,          # Positional encoding dimensions
        temporal_encoding_dim: int = 6,  # Temporal encoding dimensions
        hidden_dim: int = 256,          # Size of hidden layers
        num_layers: int = 6,            # Number of hidden layers
        skip_connections: List[int] = [3],  # Layers with skip connections
        use_attention: bool = True      # Whether to use attention mechanism
    ):
        super(NeuralCorrespondenceField, self).__init__()
        
        self.encoding_dim = encoding_dim
        self.temporal_encoding_dim = temporal_encoding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.use_attention = use_attention
        
        # Input dimension after positional encoding
        pos_enc_dim = 3 * (2 * encoding_dim + 1)  # Spatial encoding
        time_enc_dim = 1 * (2 * temporal_encoding_dim + 1)  # Temporal encoding
        input_dim = pos_enc_dim + time_enc_dim
        
        # Network architecture
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for i in range(1, num_layers):
            if i in skip_connections:
                self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Motion vector output (3D displacement + confidence)
        self.motion_layer = nn.Linear(hidden_dim, 4)
        
        # Attention mechanism for temporal dynamics
        if use_attention:
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.attention_output = nn.Linear(hidden_dim, hidden_dim)
        
    def positional_encoding(self, x: torch.Tensor, encoding_dim: int) -> torch.Tensor:
        """Apply positional encoding to spatial or temporal coordinates"""
        # x shape: (batch_size, n_dims)
        frequencies = 2.0 ** torch.arange(0, encoding_dim, device=x.device).float()
        
        # Reshape frequencies for broadcasting
        frequencies = frequencies.view(1, 1, -1)
        
        # Reshape input for broadcasting
        x_reshaped = x.unsqueeze(-1)
        
        # Compute sin and cos encodings
        x_frequencies = x_reshaped * frequencies
        sin_encodings = torch.sin(x_frequencies)
        cos_encodings = torch.cos(x_frequencies)
        
        # Concatenate original input with encodings
        encodings = torch.cat([x, sin_encodings.reshape(x.shape[0], -1), 
                              cos_encodings.reshape(x.shape[0], -1)], dim=-1)
        return encodings
    
    def attention_block(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head attention mechanism for temporal correlation"""
        if not self.use_attention:
            return x
        
        batch_size = x.shape[0]
        
        # Project to query, key, value
        query = self.query_proj(x).view(batch_size, -1)
        key = self.key_proj(x).view(batch_size, -1)
        value = self.value_proj(x).view(batch_size, -1)
        
        # Simple self-attention (no multi-head in this simplified version)
        attention_scores = torch.matmul(query.unsqueeze(1), key.unsqueeze(2)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value.unsqueeze(2)).squeeze(2)
        
        # Residual connection and projection
        output = x + self.attention_output(context)
        return output
    
    def forward(
        self, 
        positions: torch.Tensor,    # (batch_size, 3) - spatial coordinates
        times: torch.Tensor         # (batch_size, 1) - time coordinates
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        # Apply positional encoding
        pos_encoded = self.positional_encoding(positions, self.encoding_dim)
        time_encoded = self.positional_encoding(times, self.temporal_encoding_dim)
        
        # Concatenate encodings
        x = torch.cat([pos_encoded, time_encoded], dim=-1)
        input_x = x  # Save for skip connections
        
        # Initial layer
        h = F.relu(self.layers[0](x))
        
        # Hidden layers with skip connections
        for i in range(1, self.num_layers):
            if i in self.skip_connections:
                h = torch.cat([h, input_x], dim=-1)
            h = F.relu(self.layers[i](h))
            
            # Apply attention at the middle layer
            if i == self.num_layers // 2 and self.use_attention:
                h = self.attention_block(h)
        
        # Output motion vectors and confidence
        motion_output = self.motion_layer(h)
        
        # Split into motion vector and confidence
        motion_vector = motion_output[:, :3]  # 3D displacement vector
        confidence = torch.sigmoid(motion_output[:, 3:4])  # Confidence score (0-1)
        
        return {
            'motion_vector': motion_vector,
            'confidence': confidence
        }


class MotionTracker:
    """
    Motion tracker using Neural Correspondence Fields
    """
    
    def __init__(
        self,
        ncf_model: NeuralCorrespondenceField,
        time_window: int = 10,           # Number of time steps to track
        distance_threshold: float = 0.1,  # Threshold for correspondence matching
        chunk_size: int = 32768          # Chunk size for batched inference
    ):
        self.model = ncf_model
        self.time_window = time_window
        self.distance_threshold = distance_threshold
        self.chunk_size = chunk_size
        
    def predict_motion(
        self,
        positions: torch.Tensor,      # (N, 3) - spatial coordinates
        times: torch.Tensor,          # (N, 1) - time coordinates 
        batch_size: int = 4096        # Batch size for inference
    ) -> Dict[str, torch.Tensor]:
        """Predict motion vectors for given positions and times"""
        n_points = positions.shape[0]
        
        # Initialize outputs
        motion_vectors = []
        confidences = []
        
        # Process in batches
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            pos_batch = positions[i:end_idx]
            time_batch = times[i:end_idx]
            
            with torch.no_grad():
                output = self.model(pos_batch, time_batch)
                
            motion_vectors.append(output['motion_vector'])
            confidences.append(output['confidence'])
        
        # Concatenate results
        motion_vectors = torch.cat(motion_vectors, dim=0)
        confidences = torch.cat(confidences, dim=0)
        
        return {
            'motion_vectors': motion_vectors,
            'confidences': confidences
        }
    
    def track_trajectory(
        self,
        initial_positions: torch.Tensor,    # (N, 3) - initial positions
        time_steps: int,                    # Number of time steps to predict
        dt: float = 0.1,                    # Time step size
        use_confidence_threshold: bool = True  # Whether to filter by confidence
    ) -> Dict[str, torch.Tensor]:
        """Track trajectories from initial positions over multiple time steps"""
        n_points = initial_positions.shape[0]
        device = initial_positions.device
        
        # Initialize trajectory storage
        trajectories = torch.zeros((time_steps, n_points, 3), device=device)
        trajectories[0] = initial_positions
        
        confidence_history = torch.zeros((time_steps, n_points, 1), device=device)
        
        # Start with current positions
        current_positions = initial_positions.clone()
        
        # Track through time
        for t in range(1, time_steps):
            # Create time tensor for current step
            current_time = torch.ones((n_points, 1), device=device) * t * dt
            
            # Predict motion
            motion_output = self.predict_motion(current_positions, current_time)
            motion_vectors = motion_output['motion_vectors']
            confidences = motion_output['confidences']
            
            # Store confidence
            confidence_history[t] = confidences
            
            # Update positions based on predicted motion
            if use_confidence_threshold:
                # Only update positions with high confidence
                mask = confidences.squeeze(-1) > 0.5
                current_positions[mask] = current_positions[mask] + motion_vectors[mask]
            else:
                # Weight updates by confidence
                current_positions = current_positions + motion_vectors * confidences
                
            # Store updated positions
            trajectories[t] = current_positions
            
        return {
            'trajectories': trajectories,                 # (time_steps, n_points, 3)
            'confidence_history': confidence_history      # (time_steps, n_points, 1)
        }
    
    def compute_correspondence_field(
        self,
        positions: torch.Tensor,      # (N, 3) - spatial coordinates
        times: List[float],            # List of time values to compute field for
        resolution: Tuple[int, int, int] = (32, 32, 32),  # Field resolution
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # Spatial bounds
    ) -> Dict[str, torch.Tensor]:
        """Compute a dense correspondence field for visualization"""
        device = positions.device
        
        # Determine bounds if not provided
        if bounds is None:
            min_bounds = torch.min(positions, dim=0)[0] - 0.1
            max_bounds = torch.max(positions, dim=0)[0] + 0.1
        else:
            min_bounds, max_bounds = bounds
        
        # Create grid
        x = torch.linspace(min_bounds[0].item(), max_bounds[0].item(), resolution[0], device=device)
        y = torch.linspace(min_bounds[1].item(), max_bounds[1].item(), resolution[1], device=device)
        z = torch.linspace(min_bounds[2].item(), max_bounds[2].item(), resolution[2], device=device)
        
        # Create meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid_positions = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        # Initialize output fields
        n_grid_points = grid_positions.shape[0]
        n_times = len(times)
        
        flow_fields = torch.zeros((n_times, *resolution, 3), device=device)
        confidence_fields = torch.zeros((n_times, *resolution), device=device)
        
        # Compute field for each time
        for t_idx, t in enumerate(times):
            # Create time tensor
            time_tensor = torch.ones((n_grid_points, 1), device=device) * t
            
            # Predict motion in batches
            all_motions = []
            all_confidences = []
            
            for i in range(0, n_grid_points, self.chunk_size):
                end_idx = min(i + self.chunk_size, n_grid_points)
                pos_batch = grid_positions[i:end_idx]
                time_batch = time_tensor[i:end_idx]
                
                with torch.no_grad():
                    output = self.model(pos_batch, time_batch)
                    
                all_motions.append(output['motion_vector'])
                all_confidences.append(output['confidence'])
            
            # Combine batches
            motion_vectors = torch.cat(all_motions, dim=0)
            confidences = torch.cat(all_confidences, dim=0)
            
            # Reshape to grid
            flow_fields[t_idx] = motion_vectors.reshape(*resolution, 3)
            confidence_fields[t_idx] = confidences.reshape(*resolution)
            
        return {
            'flow_fields': flow_fields,                # (n_times, *resolution, 3)
            'confidence_fields': confidence_fields,    # (n_times, *resolution)
            'grid_positions': grid_positions.reshape(*resolution, 3)  # (*resolution, 3)
        }


class DOMA(nn.Module):
    """
    Dynamic Object Motion Analysis (DOMA) model
    Combines NCF with object detection and tracking for RF source localization
    """
    
    def __init__(
        self,
        ncf_model: NeuralCorrespondenceField,
        feature_dim: int = 64,
        num_objects: int = 10,      # Maximum number of trackable objects
        use_transformer: bool = True  # Whether to use transformer for temporal modeling
    ):
        super(DOMA, self).__init__()
        
        self.ncf_model = ncf_model
        self.feature_dim = feature_dim
        self.num_objects = num_objects
        self.use_transformer = use_transformer
        
        # Object detection head
        self.detection_head = nn.Sequential(
            nn.Linear(self.ncf_model.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * num_objects)  # x,y,z, w,h,d, confidence for each object
        )
        
        # Object classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.ncf_model.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_objects)  # Classification score for each object type
        )
        
        # Transformer for temporal modeling (if enabled)
        if use_transformer:
            self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim, 
                nhead=4,
                dim_feedforward=2*feature_dim,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(
                self.transformer_encoder_layer, 
                num_layers=2
            )
            
        # Position and feature embedding
        self.position_embedding = nn.Linear(3, feature_dim // 2)
        self.time_embedding = nn.Linear(1, feature_dim // 2)
        
    def embed_inputs(
        self, 
        positions: torch.Tensor,  # (batch_size, 3)
        times: torch.Tensor       # (batch_size, 1)
    ) -> torch.Tensor:
        """Embed positions and times into feature space"""
        pos_embedding = self.position_embedding(positions)
        time_embedding = self.time_embedding(times)
        return torch.cat([pos_embedding, time_embedding], dim=-1)
    
    def forward(
        self, 
        positions: torch.Tensor,    # (batch_size, seq_len, 3) - spatial coordinates
        times: torch.Tensor,        # (batch_size, seq_len, 1) - time coordinates
        rf_features: Optional[torch.Tensor] = None  # Optional RF features
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        batch_size, seq_len, _ = positions.shape
        
        # Flatten batch and sequence dimensions for NCF
        flat_positions = positions.reshape(-1, 3)
        flat_times = times.reshape(-1, 1)
        
        # Get motion vectors from NCF
        ncf_output = self.ncf_model(flat_positions, flat_times)
        motion_vectors = ncf_output['motion_vector'].reshape(batch_size, seq_len, 3)
        confidence = ncf_output['confidence'].reshape(batch_size, seq_len, 1)
        
        # Extract features from NCF's hidden layers (assuming access)
        # In practice, you'd need to modify NCF to expose these
        # This is a simplified version
        embedded_features = self.embed_inputs(flat_positions, flat_times)
        embedded_features = embedded_features.reshape(batch_size, seq_len, -1)
        
        # Apply transformer for temporal modeling if enabled
        if self.use_transformer:
            # Transformer expects [seq_len, batch_size, feature_dim]
            embedded_features = embedded_features.transpose(0, 1)
            transformed_features = self.transformer(embedded_features)
            # Back to [batch_size, seq_len, feature_dim]
            transformed_features = transformed_features.transpose(0, 1)
        else:
            transformed_features = embedded_features
            
        # Predict bounding boxes for detected objects
        # Take the last timestep features for detection
        last_features = transformed_features[:, -1]
        detection_output = self.detection_head(last_features)
        
        # Reshape detection output to [batch_size, num_objects, 7]
        detection_output = detection_output.reshape(batch_size, self.num_objects, 7)
        
        # Split into box parameters and confidence
        boxes = detection_output[:, :, :6]  # x,y,z, width,height,depth
        box_confidence = torch.sigmoid(detection_output[:, :, 6:7])  # Detection confidence
        
        # Classification scores
        classification_scores = self.classification_head(last_features)
        classification_scores = classification_scores.reshape(batch_size, self.num_objects)
        
        return {
            'motion_vectors': motion_vectors,            # (batch_size, seq_len, 3)
            'motion_confidence': confidence,             # (batch_size, seq_len, 1)
            'detected_boxes': boxes,                     # (batch_size, num_objects, 6)
            'box_confidence': box_confidence,            # (batch_size, num_objects, 1)
            'classification_scores': classification_scores  # (batch_size, num_objects)
        }
        
    def detect_rf_sources(
        self,
        rf_data: torch.Tensor,       # (batch_size, seq_len, height, width, depth, features)
        time_steps: torch.Tensor     # (batch_size, seq_len, 1)
    ) -> Dict[str, torch.Tensor]:
        """Detect and track RF sources in volumetric data"""
        batch_size, seq_len, height, width, depth, _ = rf_data.shape
        
        # Convert grid to point cloud for processing
        # Create coordinate grid
        x = torch.linspace(-1, 1, height, device=rf_data.device)
        y = torch.linspace(-1, 1, width, device=rf_data.device)
        z = torch.linspace(-1, 1, depth, device=rf_data.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        # Expand grid to batch and sequence dimensions
        grid_coords = grid_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1, -1, -1)
        
        # Extract points with significant RF activity
        # Use the first feature dimension as signal strength
        signal_strength = rf_data[..., 0]
        
        # Find points above threshold (adapt threshold based on data)
        threshold = torch.mean(signal_strength) + torch.std(signal_strength)
        active_points_mask = signal_strength > threshold
        
        # Process each sequence in the batch
        all_trajectories = []
        all_confidences = []
        
        for b in range(batch_size):
            trajectories = []
            confidences = []
            
            for s in range(seq_len):
                # Get coordinates of active points
                active_coords = grid_coords[b, s][active_points_mask[b, s]]
                
                # If no active points, skip
                if active_coords.shape[0] == 0:
                    trajectories.append(torch.zeros((0, 3), device=rf_data.device))
                    confidences.append(torch.zeros((0, 1), device=rf_data.device))
                    continue
                
                # Get corresponding RF features
                active_features = rf_data[b, s][active_points_mask[b, s]]
                
                # Create time tensor
                time_tensor = time_steps[b, s].expand(active_coords.shape[0], 1)
                
                # Get motion prediction from NCF
                with torch.no_grad():
                    output = self.ncf_model(active_coords, time_tensor)
                
                trajectories.append(active_coords)
                confidences.append(output['confidence'])
            
            all_trajectories.append(trajectories)
            all_confidences.append(confidences)
        
        # Process detection for the batch
        # Use the last timestep for final detection
        final_positions = []
        final_times = []
        
        for b in range(batch_size):
            if len(all_trajectories[b]) > 0 and all_trajectories[b][-1].shape[0] > 0:
                final_positions.append(all_trajectories[b][-1])
                final_times.append(time_steps[b, -1].expand(all_trajectories[b][-1].shape[0], 1))
            else:
                # Create a dummy point if no active points
                final_positions.append(torch.zeros((1, 3), device=rf_data.device))
                final_times.append(time_steps[b, -1].expand(1, 1))
                
        # Get detections
        with torch.no_grad():
            # Process each sample individually since they may have different numbers of points
            detections = []
            
            for pos, time in zip(final_positions, final_times):
                # Embed positions and times
                embedded = self.embed_inputs(pos, time)
                
                # Get mean representation for detection
                mean_embedding = torch.mean(embedded, dim=0, keepdim=True)
                
                # Get detection output
                detection_output = self.detection_head(mean_embedding)
                detection_output = detection_output.reshape(1, self.num_objects, 7)
                
                # Split into box parameters and confidence
                boxes = detection_output[:, :, :6]  # x,y,z, width,height,depth
                box_confidence = torch.sigmoid(detection_output[:, :, 6:7])
                
                # Only keep high-confidence detections
                valid_mask = box_confidence.squeeze(-1) > 0.5
                valid_boxes = boxes[:, valid_mask.squeeze()]
                valid_confidence = box_confidence[:, valid_mask.squeeze()]
                
                detections.append((valid_boxes, valid_confidence))
        
        return {
            'trajectories': all_trajectories,  # List of trajectories for each sequence
            'confidences': all_confidences,    # List of confidence values
            'detections': detections           # List of (boxes, confidence) tuples
        }