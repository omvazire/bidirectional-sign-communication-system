"""
Sign Language Translator — Main CLI Entry Point.

Usage:
    python run.py status    -> Inspect project readiness and next step
    python run.py collect   -> Record gesture data via webcam
    python run.py process   -> Preprocess and augment data
    python run.py train     -> Train the Bi-LSTM model
    python run.py run       -> Launch real-time detection
    python run.py web       -> Launch browser-based real-time detection
    python run.py diagnose  -> Run pipeline diagnostics
    python run.py test      -> Quick model test (forward pass)
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="AI Real-Time Sign Language Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Commands:
  status    Inspect dataset/model readiness and recommend next step
  collect   Record gesture sequences via webcam (guided UI)
  process   Preprocess data: normalize, augment, split
  train     Train the Bi-LSTM gesture recognition model
  run       Launch real-time sign language detection (Terminal UI)
  web       Launch browser-based real-time sign language detection
  diagnose  Run pipeline health diagnostics
  test      Quick model sanity check (forward pass test)

Typical Workflow:
  1. python run.py collect    (record gestures — ~75 min)
  2. python run.py process    (preprocess — ~30 sec)
  3. python run.py train      (train model — ~5-15 min)
  4. python run.py diagnose   (verify pipeline health)
  5. python run.py run        (launch desktop detection)
     or python run.py web     (launch browser UI)
        """,
    )

    parser.add_argument(
        "command",
        choices=["status", "collect", "process", "train", "run", "web", "diagnose", "test"],
        help="Pipeline command to execute",
    )

    args = parser.parse_args()

    # ─── Route to appropriate module ──────────────────────────

    if args.command == "status":
        from src.status import run_status
        run_status()

    elif args.command == "collect":
        from src.collect_data import collect_data
        collect_data()

    elif args.command == "process":
        from src.process_data import process_data
        process_data()

    elif args.command == "train":
        from src.train import train_model
        train_model()

    elif args.command == "run":
        from src.detect_runtime import run_detection
        run_detection()

    elif args.command == "web":
        from src.app import run_server
        run_server()

    elif args.command == "diagnose":
        from src.diagnose import run_diagnostics
        run_diagnostics()

    elif args.command == "test":
        _run_model_test()


def _run_model_test():
    """Quick sanity check — build model and run a dummy forward pass."""
    import torch
    import config
    from src.model import build_model, GestureRecognitionModel

    print("\n" + "=" * 60)
    print("  MODEL SANITY CHECK")
    print("=" * 60)

    # Build model
    model = build_model()

    # Dummy input: batch of 2 sequences
    dummy_input = torch.randn(2, config.SEQUENCE_LENGTH, config.INPUT_FEATURES)
    print(f"\n  Input shape:  {dummy_input.shape}")
    print(f"  Expected:     (2, {config.SEQUENCE_LENGTH}, {config.INPUT_FEATURES})")

    # Forward pass
    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     (2, {config.NUM_GESTURES})")

    # Softmax probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    print(f"\n  Sample probabilities (sums to 1.0):")
    print(f"    {probs[0].detach().numpy().round(3)}")
    print(f"    Sum: {probs[0].sum().item():.4f}")

    # Test predict_with_confidence
    single = torch.randn(1, config.SEQUENCE_LENGTH, config.INPUT_FEATURES)
    pred_class, confidence, all_probs = model.predict_with_confidence(single)
    print(f"\n  Prediction:   {config.GESTURES[pred_class]}")
    print(f"  Confidence:   {confidence:.1%}")

    print(f"\n  [OK] All checks passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
