#!/usr/bin/env python3
"""
Example scripts showing how to use the YAML configuration file with the enhanced CNN training script.

This file demonstrates different ways to use training_config.yaml for configuration management.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_with_default_config():
    """Run training with default YAML configuration"""
    print("🚀 Running with default YAML configuration (training_config.yaml)")
    print("=" * 70)
    
    cmd = [
        "uv", "run", "train_local_sonnet.py",
        # No need to specify config file - it will automatically load training_config.yaml
        "--epochs", "20",  # Override epochs from command line
        "--stream-artifacts"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_with_custom_config():
    """Run training with a custom configuration file"""
    print("\n🔧 Running with custom configuration file")
    print("=" * 70)
    
    # First, let's create a custom config for faster training
    custom_config = """
# Custom Configuration for Quick Training
model:
  name: "resnet50"  # Smaller model for faster training
  pretrained: true
  num_classes: 2

data:
  image_size: 192  # Smaller images for faster training
  batch_size: 64   # Larger batch size
  num_workers: 4
  use_weighted_sampling: true

training:
  epochs: 10       # Fewer epochs for testing
  learning_rate: 0.01
  weight_decay: 0.0001
  scheduler: "onecycle"
  patience: 5

optimizer:
  type: "adamw"
  backbone_lr_multiplier: 0.1
  eps: 1e-8

augmentation:
  train:
    horizontal_flip: 0.5
    vertical_flip: 0.0     # Disable vertical flip
    rotation_degrees: 10   # Less rotation
    color_jitter:
      brightness: 0.1
      contrast: 0.1
      saturation: 0.1
      hue: 0.05
    gaussian_blur_prob: 0.2
    random_erasing_prob: 0.2

clearml:
  project_name: "Quick-CNN-Test"
  task_name: "fast_training_test"
  stream_artifacts: true
"""
    
    # Save custom config
    with open('custom_training_config.yaml', 'w') as f:
        f.write(custom_config)
    
    cmd = [
        "uv", "run", "train_local_sonnet.py",
        "--config", "custom_training_config.yaml",
        "--name", "custom_config_run"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_with_config_override():
    """Run training showing how command line args override config"""
    print("\n⚡ Running with config + command line overrides")
    print("=" * 70)
    print("This shows how command line arguments take precedence over YAML config")
    
    cmd = [
        "uv", "run", "train_local_sonnet.py",
        # These will override the YAML config values
        "--lr", "0.005",           # Override learning rate
        "--batch", "16",           # Override batch size
        "--epochs", "15",          # Override epochs
        "--model-name", "resnet101", # Override model
        "--scheduler", "plateau",   # Override scheduler
        "--patience", "7",         # Override patience
        "--name", "config_override_run"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Note: These command line values will override the YAML configuration")
    subprocess.run(cmd)

def show_config_structure():
    """Display the structure of the configuration file"""
    print("\n📋 YAML Configuration File Structure")
    print("=" * 70)
    
    structure = """
The training_config.yaml file has the following structure:

📁 model:                    # Model architecture settings
   ├── name                  # Model type (resnet50/101/152)
   ├── pretrained           # Use pretrained weights
   └── num_classes          # Number of output classes

📁 data:                     # Data loading settings
   ├── image_size           # Input image size
   ├── batch_size           # Training batch size
   ├── num_workers          # Data loader workers
   └── use_weighted_sampling # Handle class imbalance

📁 augmentation:             # Data augmentation settings
   └── train:               # Training augmentations
       ├── horizontal_flip  # Flip probability
       ├── vertical_flip    # Vertical flip probability
       ├── rotation_degrees # Max rotation angle
       ├── color_jitter     # Color augmentation
       ├── gaussian_blur_prob # Blur probability
       └── random_erasing_prob # Erasing probability

📁 training:                 # Training hyperparameters
   ├── epochs               # Number of training epochs
   ├── learning_rate        # Base learning rate
   ├── weight_decay         # L2 regularization
   ├── scheduler            # LR scheduler type
   ├── patience             # Early stopping patience
   └── min_delta            # Minimum improvement

📁 optimizer:                # Optimizer configuration
   ├── type                 # Optimizer type (adamw/sgd)
   ├── backbone_lr_multiplier # LR factor for backbone
   └── eps                  # Optimizer epsilon

📁 clearml:                  # ClearML integration
   ├── project_name         # ClearML project
   ├── task_name            # Task name
   └── stream_artifacts     # Real-time upload

📁 hardware:                 # Hardware optimization
   ├── mixed_precision      # Use mixed precision
   ├── pin_memory           # Pin memory for GPU
   └── persistent_workers   # Keep workers alive

📁 search_ranges:            # For hyperparameter optimization
   ├── learning_rate        # Search range for LR
   ├── weight_decay         # Search range for weight decay
   └── batch_size           # Batch size options
"""
    print(structure)

def create_production_config():
    """Create a production-ready configuration"""
    print("\n🏭 Creating production configuration")
    print("=" * 70)
    
    production_config = """
# Production Configuration for Best Performance
model:
  name: "resnet152"
  pretrained: true
  num_classes: 2

data:
  image_size: 256
  batch_size: 32
  num_workers: 8
  use_weighted_sampling: true

augmentation:
  train:
    random_crop: true
    horizontal_flip: 0.5
    vertical_flip: 0.1
    rotation_degrees: 20
    color_jitter:
      brightness: 0.3
      contrast: 0.3
      saturation: 0.3
      hue: 0.15
    gaussian_blur_prob: 0.4
    random_erasing_prob: 0.4

training:
  epochs: 100
  learning_rate: 0.0005
  weight_decay: 0.0001
  scheduler: "cosine"
  patience: 15
  min_delta: 0.0005

optimizer:
  type: "adamw"
  backbone_lr_multiplier: 0.05
  eps: 1e-8

loss:
  type: "crossentropy"
  label_smoothing: 0.15

regularization:
  dropout_rates: [0.6, 0.4, 0.3]
  gradient_clip_norm: 1.0

validation:
  test_time_augmentation: true
  cross_validation: false

clearml:
  project_name: "Production-CNN-Training"
  task_name: "production_resnet_training"
  stream_artifacts: true

hardware:
  mixed_precision: true
  pin_memory: true
  persistent_workers: true
"""
    
    with open('production_config.yaml', 'w') as f:
        f.write(production_config)
    
    print("✅ Created production_config.yaml")
    print("To use it: uv run train_local_sonnet.py --config production_config.yaml")

def main():
    """Main function to demonstrate configuration usage"""
    print("🎯 Enhanced CNN Training - YAML Configuration Examples")
    print("=" * 70)
    
    print("Available examples:")
    print("1. Show configuration file structure")
    print("2. Create production configuration")
    print("3. Run with default config")
    print("4. Run with custom config")
    print("5. Run with config + command line overrides")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-6): ").strip()
            
            if choice == '1':
                show_config_structure()
            elif choice == '2':
                create_production_config()
            elif choice == '3':
                run_with_default_config()
            elif choice == '4':
                run_with_custom_config()
            elif choice == '5':
                run_with_config_override()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
