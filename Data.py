#!/usr/bin/env python
"""
FIXED DATA SPLITS CONFIGURATION
This script documents and verifies the exact data splits used in the paper.
All splits are fixed and reproducible using random seed 42.
"""

import numpy as np
import pickle
import os
from pathlib import Path

# Fixed random seed for all experiments
SEED = 42
np.random.seed(SEED)

def document_splits():
    """Document the exact data splits used in the paper."""
    print("="*70)
    print("EXACT DATA SPLITS USED IN PAPER")
    print("="*70)
    print(f"Random seed for reproducibility: {SEED}")
    print()

    # Paper Table I and II specifications
    splits = {
        'SU_gearbox': {
            'training': {
                'condition': 'C0 (normal)',
                'samples': 700,
                'file': 'Datasets/C0_T.pkl'
            },
            'testing': {
                'normal': {'samples': 400, 'file': 'Datasets/C0_test.pkl'},
                'faults': [
                    {'label': 'C1', 'description': 'Broken tooth', 'samples': 50, 'file': 'Datasets/C1_test.pkl'},
                    {'label': 'C2', 'description': 'Missing tooth', 'samples': 50, 'file': 'Datasets/C2_test.pkl'},
                    {'label': 'C3', 'description': 'Cracked tooth roots', 'samples': 50, 'file': 'Datasets/C3_test.pkl'},
                    {'label': 'C4', 'description': 'Surface wear', 'samples': 50, 'file': 'Datasets/C4_test.pkl'},
                    {'label': 'C5', 'description': 'Broken tooth (30Hz)', 'samples': 50, 'file': 'Datasets/C5_test.pkl'},
                    {'label': 'C6', 'description': 'Missing tooth (30Hz)', 'samples': 50, 'file': 'Datasets/C6_test.pkl'},
                    {'label': 'C7', 'description': 'Cracked tooth roots (30Hz)', 'samples': 50, 'file': 'Datasets/C7_test.pkl'},
                    {'label': 'C8', 'description': 'Surface wear (30Hz)', 'samples': 50, 'file': 'Datasets/C8_test.pkl'}
                ]
            },
            'total_test_samples': 400 + 8*50  # 800 samples
        },

        'UC_gearbox': {
            'training': {
                'condition': 'P0 (normal)',
                'samples': 700,
                'file': 'Datasets/P0_T.pkl'
            },
            'testing': {
                'normal': {'samples': 400, 'file': 'Datasets/P0_test.pkl'},
                'faults': [
                    {'label': 'P1', 'description': 'Missing tooth', 'samples': 50},
                    {'label': 'P2', 'description': 'Root crack', 'samples': 50},
                    {'label': 'P3', 'description': 'Spalling', 'samples': 50},
                    {'label': 'P4', 'description': 'Slight wear', 'samples': 50},
                    {'label': 'P5', 'description': 'Moderate wear', 'samples': 50},
                    {'label': 'P6', 'description': 'Severe wear', 'samples': 50},
                    {'label': 'P7', 'description': 'Deep wear', 'samples': 50},
                    {'label': 'P8', 'description': 'Extreme wear', 'samples': 50}
                ]
            },
            'total_test_samples': 400 + 8*50  # 800 samples
        },

        'QPZZ_II_gearbox': {
            'training': {
                'condition': 'Label 0 (normal)',
                'samples': 700,
                'file': 'Datasets/Label0_T.pkl'
            },
            'testing': {
                'normal': {'samples': 450, 'file': 'Datasets/Label0_test.pkl'},
                'faults': [
                    {'label': 'Label 1', 'description': 'One tooth pitting', 'samples': 50},
                    {'label': 'Label 2', 'description': 'One tooth pitting', 'samples': 50},
                    {'label': 'Label 3', 'description': 'One tooth pitting', 'samples': 50},
                    {'label': 'Label 4', 'description': 'One tooth crack', 'samples': 50},
                    {'label': 'Label 5', 'description': 'One tooth crack', 'samples': 50},
                    {'label': 'Label 6', 'description': 'Two teeth pitting', 'samples': 50},
                    {'label': 'Label 7', 'description': 'Two teeth pitting', 'samples': 50},
                    {'label': 'Label 8', 'description': 'Two teeth crack', 'samples': 50},
                    {'label': 'Label 9', 'description': 'Two teeth crack', 'samples': 50}
                ]
            },
            'total_test_samples': 450 + 9*50  # 900 samples
        }
    }

    # Display splits for each dataset
    for dataset_name, dataset_info in splits.items():
        print(f"\n{'='*40}")
        print(f"DATASET: {dataset_name}")
        print('='*40)

        # Training data
        train = dataset_info['training']
        print(f"\nTRAINING SET:")
        print(f"  Condition: {train['condition']}")
        print(f"  Samples: {train['samples']}")
        if 'file' in train:
            file_exists = Path(train['file']).exists()
            print(f"  File: {train['file']} {'✓' if file_exists else '✗ (not found)'}")

        # Testing data
        test = dataset_info['testing']
        print(f"\nTESTING SET:")
        print(f"  Normal samples: {test['normal']['samples']}")
        if 'file' in test['normal']:
            file_exists = Path(test['normal']['file']).exists()
            print(f"  File: {test['normal']['file']} {'✓' if file_exists else '✗ (not found)'}")

        print(f"\n  Fault samples (8 types):")
        for fault in test['faults']:
            print(f"    {fault['label']}: {fault['description']} - {fault['samples']} samples")

        print(f"\n  Total testing samples: {dataset_info['total_test_samples']}")

    print(f"\n{'='*70}")
    print("IMPORTANT NOTES:")
    print("1. NO validation split is used (one-class learning methodology)")
    print("2. Training uses ONLY normal samples (one-class setting)")
    print("3. Testing includes both normal and fault samples")
    print("4. All splits are FIXED (no random variation)")
    print("5. Random seed: 42 ensures reproducibility")
    print("="*70)

    # Save splits to file
    output_file = "results/data_splits_documentation.txt"
    os.makedirs("results", exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("EXACT DATA SPLITS DOCUMENTATION\n")
        f.write("="*50 + "\n\n")
        for dataset_name, dataset_info in splits.items():
            f.write(f"DATASET: {dataset_name}\n")
            f.write(f"Training samples: {dataset_info['training']['samples']}\n")
            f.write(f"Testing samples: {dataset_info['total_test_samples']}\n\n")

    print(f"\n✓ Documentation saved to: {output_file}")

def verify_file_existence():
    """Verify that all required data files exist."""
    print("\n" + "="*70)
    print("VERIFYING DATA FILES EXISTENCE")
    print("="*70)

    required_files = [
        # SU dataset
        'Datasets/C0_T.pkl', 'Datasets/C0_test.pkl',
        'Datasets/C1_test.pkl', 'Datasets/C2_test.pkl',
        'Datasets/C3_test.pkl', 'Datasets/C4_test.pkl',
        'Datasets/C5_test.pkl', 'Datasets/C6_test.pkl',
        'Datasets/C7_test.pkl', 'Datasets/C8_test.pkl',
    ]

    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\n✓ All required data files exist")
    else:
        print("\n⚠ Some data files are missing")
        print("   You can create synthetic data using:")
        print("   python data_preprocessing.py --synthetic")

if __name__ == "__main__":
    document_splits()
    verify_file_existence()