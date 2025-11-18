#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for inference/prediction on new queries

Usage:
    python scripts/run_inference.py --checkpoint checkpoints/mode/best.pt --query "What is 2+2?"
    python scripts/run_inference.py --checkpoint checkpoints/mode/best.pt --input_file queries.txt
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mode import MoDEInference


def main():
    parser = argparse.ArgumentParser(description="MoDE Budget Model Inference")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to predict budget for"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="File with queries (one per line or JSON)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for predictions (JSON)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Return detailed predictions with expert weights"
    )
    
    args = parser.parse_args()
    
    if not args.query and not args.input_file:
        parser.error("Must provide either --query or --input_file")
    
    print("="*60)
    print("MoDE Budget Model Inference")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    inference = MoDEInference(args.checkpoint)
    
    # Prepare queries
    queries = []
    if args.query:
        queries = [args.query]
    elif args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries = data
                elif isinstance(data, dict) and 'queries' in data:
                    queries = data['queries']
                else:
                    queries = [str(data)]
        else:
            with open(input_path, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
    
    # Predict
    print(f"\nPredicting budgets for {len(queries)} queries...")
    results = []
    for query in queries:
        result = inference.predict(query, return_detailed=args.detailed)
        result['query'] = query
        results.append(result)
        
        # Print result
        print(f"\nQuery: {query}")
        print(f"  Predicted Budget: {result['budget']} tokens")
        print(f"  Difficulty: {result['difficulty']}")
        print(f"  Dominant Expert: {result['dominant_expert']}")
        print(f"  Is Unlimited: {result['is_unlimited']}")
        
        if args.detailed:
            print(f"  Expert Weights: {result.get('expert_weights', {})}")
            print(f"  Expert Budgets: {result.get('expert_budgets', {})}")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved predictions to {output_path}")
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

