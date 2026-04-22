"""
Model Architecture — Bi-LSTM with Self-Attention for Gesture Classification.

Architecture:
    Input (25, 126) -> LayerNorm -> Bi-LSTM (2 layers) -> Self-Attention -> FC -> Softmax

The attention mechanism lets the model focus on the most discriminative frames
in a gesture sequence rather than relying solely on the final hidden state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class SelfAttention(nn.Module):
    """
    Self-attention layer that learns which timesteps are most important.

    Instead of using just the last hidden state of the LSTM, attention
    computes a weighted sum of ALL timestep outputs, giving higher weight
    to the most informative frames.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — attention-weighted representation
            weights: (batch, seq_len) — attention weights for visualization
        """
        # Compute attention scores for each timestep
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)                # (batch, seq_len)

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output            # (batch, seq_len, hidden_size)
        ).squeeze(1)              # (batch, hidden_size)

        return context, weights


class GestureRecognitionModel(nn.Module):
    """
    Bi-LSTM + Self-Attention model for sign language gesture recognition.

    Architecture:
        1. LayerNorm — stabilize input distribution
        2. Bi-LSTM (2 layers) — capture temporal patterns in both directions
        3. Self-Attention — focus on most discriminative frames
        4. FC layers — classification head with BatchNorm + Dropout

    Args:
        input_size:  Number of features per frame (default: 63)
        hidden_size: LSTM hidden units (default: 128)
        num_layers:  Stacked LSTM layers (default: 2)
        num_classes: Number of gesture classes (from config)
        dropout:     Dropout rate (default: 0.4)
    """

    def __init__(
        self,
        input_size=config.INPUT_FEATURES,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_GESTURES,
        dropout=config.DROPOUT,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalization
        self.layer_norm = nn.LayerNorm(input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention over timesteps
        # Bi-LSTM output is 2 * hidden_size (forward + backward)
        self.attention = SelfAttention(hidden_size * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.67),  # Lighter dropout in later layers
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.33),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_size) — sequence of landmark features
        Returns:
            logits: (batch, num_classes) — raw class scores
        """
        # Normalize input features
        x = self.layer_norm(x)

        # Bi-LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)

        # Attention pooling
        context, _ = self.attention(lstm_out)  # (batch, hidden_size * 2)

        # Classification
        logits = self.classifier(context)  # (batch, num_classes)

        return logits

    def predict_with_confidence(self, x):
        """
        Get prediction and confidence score for real-time detection.

        Args:
            x: (1, seq_len, input_size) — single sequence
        Returns:
            predicted_class: int
            confidence: float (0.0 to 1.0)
            probabilities: np.ndarray of all class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        return (
            predicted.item(),
            confidence.item(),
            probabilities.cpu().numpy().flatten(),
        )

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(device=None):
    """
    Build and return the model, optionally moving to specified device.

    Args:
        device: torch.device (if None, stays on CPU)
    Returns:
        model: GestureRecognitionModel
    """
    model = GestureRecognitionModel()
    if device:
        model = model.to(device)

    total_params = model.count_parameters()
    print(f"[OK] Model built: {total_params:,} trainable parameters")

    return model


def load_model(model_path=None, device=None):
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to .pth file (default from config)
        device: torch.device
    Returns:
        model: Loaded GestureRecognitionModel in eval mode
    """
    if model_path is None:
        model_path = config.MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python run.py train' first."
        )

    model = GestureRecognitionModel()
    checkpoint = torch.load(model_path, map_location=device or "cpu", weights_only=True)

    # Handle both full checkpoint and state_dict-only saves
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] Model loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"acc {checkpoint.get('val_accuracy', 0):.1%})")
    else:
        model.load_state_dict(checkpoint)
        print(f"[OK] Model loaded from {model_path}")

    if device:
        model = model.to(device)

    model.eval()
    return model


if __name__ == "__main__":
    # Quick test: build model and run dummy forward pass
    model = build_model()
    dummy_input = torch.randn(2, config.SEQUENCE_LENGTH, config.INPUT_FEATURES)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (softmax): {F.softmax(output, dim=1)}")
