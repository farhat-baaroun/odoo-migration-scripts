#!/usr/bin/env python3
"""
Normalize declarations_report JSON to deterministic structure for diffing.

This script converts the output from module_report.py into a normalized format
that preserves declarations exactly as reported, without semantic inference.

CRITICAL: This script MUST be lossless - it only normalizes structure, never
infers runtime behavior or merges inheritance.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_model_key(model: Dict[str, Any]) -> str:
    """
    Get collision-safe unique key for a model.
    
    Priority:
    1. model_name if defined (e.g., "res.partner")
    2. Otherwise: "module::class_name" (e.g., "fsf_base::DocumentFolder")
    
    This ensures:
    - No overwrite collisions
    - Extension models preserved separately
    - Full traceability to source code
    """
    model_name = model.get('model_name')
    if model_name:
        return model_name
    
    # For Extension models or models without _name, use module::class_name
    module = model.get('module', 'unknown')
    class_name = model.get('class_name', 'unknown')
    return f"{module}::{class_name}"


def collect_declared_fields(model: Dict[str, Any]) -> List[str]:
    """
    Collect ONLY fields declared in this model class.
    
    Does NOT merge fields from parent models.
    This is a lossless operation - preserves exactly what was declared.
    """
    fields = []
    
    for field_list in ['regular_fields', 'computed_stored', 'computed_non_stored', 'related_fields']:
        for field in model.get(field_list, []):
            fields.append(field['name'])
    
    # Return sorted for deterministic output
    return sorted(fields)


def collect_declared_methods(model: Dict[str, Any]) -> List[str]:
    """
    Collect ONLY methods declared in this model class.
    
    Does NOT merge methods from parent models.
    This is a lossless operation - preserves exactly what was declared.
    """
    methods = []
    
    for method in model.get('methods', []):
        methods.append(method['name'])
    
    # Return sorted for deterministic output
    return sorted(methods)


def normalize_report(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize a declarations_report JSON to deterministic structure.
    
    This function is LOSSLESS - it preserves declarations exactly as reported
    without any semantic inference or inheritance resolution.
    
    Args:
        report: JSON report from module_report.py
        
    Returns:
        Dictionary mapping model_key -> normalized model dict with:
        - declared_fields: List[str] - only fields declared in this class
        - declared_methods: List[str] - only methods declared in this class
        - inherit: List[str] - inheritance metadata (preserved as-is)
        - model_type: str - classification
        - module: str - module name
        - class_name: str - Python class name
        - model_name: Optional[str] - Odoo model name (_name attribute)
    """
    models = report.get('models', [])
    
    if not models:
        return {}
    
    normalized = {}
    
    for model in models:
        model_key = get_model_key(model)
        
        normalized[model_key] = {
            "declared_fields": collect_declared_fields(model),
            "declared_methods": collect_declared_methods(model),
            "inherit": sorted(model.get('inherit', [])),
            "model_type": model.get('model_type', 'unknown'),
            "module": model.get('module', 'unknown'),
            "class_name": model.get('class_name', 'unknown'),
            "model_name": model.get('model_name'),  # Can be None for Extension models
            "file": model.get('file', 'unknown'),
            "line_number": model.get('line_number')
        }
    
    # Sort by key for deterministic output
    return dict(sorted(normalized.items()))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Normalize declarations_report JSON for deterministic diffing (lossless)"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSON report from module_report.py'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output normalized JSON file (default: input_file with _normalized suffix)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load input report
    print(f"Loading report from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # Normalize (lossless operation)
    print("Normalizing report (lossless - preserving declarations only)...")
    normalized = normalize_report(report)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_normalized.json"
    
    # Save normalized report
    output_data = {
        "source_report": str(input_path),
        "timestamp": report.get('timestamp'),
        "analyzed_path": report.get('analyzed_path'),
        "normalized_models": normalized
    }
    
    print(f"Writing normalized report to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Normalized {len(normalized)} models (lossless - declarations only)")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
