#!/usr/bin/env python3
"""
Extract API surface (fields + methods) from Odoo 14 standard models used in custom code.

This script filters the normalized Odoo 14 standard report to only include models
that are used in custom code, providing the potential API surface that custom code
might access.

ARCHITECTURAL NOTES:
====================

This script produces a CANDIDATE SET, not actual usage:
- If a standard model is referenced, ALL its declared APIs are included
- This is intentionally over-approximate
- Actual usage is verified later by usage_report.py via AST analysis

This script does NOT:
- Resolve inheritance (preserves declared APIs only)
- Infer runtime behavior
- Guess which APIs are actually used

This is correct - it provides the "possible API surface" that custom code
could access, which is then filtered by actual AST-detected usage.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set, Any


def extract_api_surface(
    matching_models: Dict[str, Dict[str, Any]],
    normalized_s14: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract API surface from Odoo 14 standard models that are used in custom code.
    
    Args:
        matching_models: Output from standard_models_used_in_custom.py
        normalized_s14: Normalized Odoo 14 standard report
        
    Returns:
        Dictionary mapping model_name -> API surface (fields + methods)
    """
    api_surface = {}
    
    # Get list of model names that match
    matching_model_names = set(matching_models.get('matches', {}).keys())
    
    # Extract API surface for matching models
    s14_models = normalized_s14.get('normalized_models', {})
    
    for model_name in matching_model_names:
        if model_name in s14_models:
            model_data = s14_models[model_name]
            api_surface[model_name] = {
                "declared_fields": model_data.get('declared_fields', []),
                "declared_methods": model_data.get('declared_methods', []),
                "inherit": model_data.get('inherit', []),
                "module": model_data.get('module', 'unknown'),
                "model_type": model_data.get('model_type', 'unknown')
            }
    
    # Sort for deterministic output
    return dict(sorted(api_surface.items()))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract API surface from Odoo 14 standard models used in custom code"
    )
    parser.add_argument(
        'matching_models',
        type=str,
        help='Path to JSON output from standard_models_used_in_custom.py'
    )
    parser.add_argument(
        'normalized_s14',
        type=str,
        help='Path to normalized Odoo 14 standard JSON report'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output JSON file (default: standv14_api_surface.json)'
    )
    
    args = parser.parse_args()
    
    # Load input files
    matching_path = Path(args.matching_models)
    s14_path = Path(args.normalized_s14)
    
    if not matching_path.exists():
        print(f"Error: Matching models file not found: {matching_path}", file=sys.stderr)
        sys.exit(1)
    
    if not s14_path.exists():
        print(f"Error: Normalized S14 report not found: {s14_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading matching models from {matching_path}...")
    with open(matching_path, 'r', encoding='utf-8') as f:
        matching_data = json.load(f)
    
    print(f"Loading normalized Odoo 14 report from {s14_path}...")
    with open(s14_path, 'r', encoding='utf-8') as f:
        s14_data = json.load(f)
    
    # Extract API surface
    print("Extracting API surface...")
    api_surface = extract_api_surface(matching_data, s14_data)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('addons_reports') / 'standv14_api_surface.json'
        output_path.parent.mkdir(exist_ok=True)
    
    # Prepare output
    output_data = {
        "source_matching_models": str(matching_path),
        "source_normalized_s14": str(s14_path),
        "total_models": len(api_surface),
        "api_surface": api_surface
    }
    
    # Calculate statistics
    total_fields = sum(len(m.get('declared_fields', [])) for m in api_surface.values())
    total_methods = sum(len(m.get('declared_methods', [])) for m in api_surface.values())
    
    print(f"Extracted API surface for {len(api_surface)} models")
    print(f"Total fields: {total_fields}, Total methods: {total_methods}")
    print(f"Writing results to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()

