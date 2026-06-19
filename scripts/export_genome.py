# export genome for the frontend

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch


def load_genome(genome_path):
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    if isinstance(genome, dict) and "genome" in genome:
        genome = genome["genome"]
    
    return genome


def genome_to_json(genome) -> dict:
    return {
        "nodes":
        {
            "ids": genome.nodes.node_ids.tolist(),
            "types": genome.nodes.node_types.tolist(),
            "activations": genome.nodes.node_activations.tolist(),
        },
        "connections": {
            "indices": genome.connections.conn_indices.tolist(),
            "weights": genome.connections.conn_weights.tolist(),
            "enabled": [bool(v) for v in genome.connections.conn_enabled.tolist()],
        }
    }     

def main():
    parser = argparse.ArgumentParser(description="Export a genome to JSON format for the frontend")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the genome checkpoint file")
    parser.add_argument("--out-dir", type=str, default="frontend/public/genomes", help="Directory to save the exported genome JSON file")
    parser.add_argument("--label", type=str, default="genome", help="Label for the exported genome JSON file")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    genome = load_genome(args.checkpoint)
    label = args.label or os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(args.out_dir, f"{label}.json")

    with open(out_path, "w") as f:
        json.dump(genome_to_json(genome), f, indent=4)
    
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = []

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    
    manifest = [m for m in manifest if m["label"] != label]  # remove any existing entry with the same label
    manifest.append({"label": label, "path": f"{label}.json"})  #

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    
    print(f"Exported {label} -> {out_path}")

if __name__ == "__main__":
    main()