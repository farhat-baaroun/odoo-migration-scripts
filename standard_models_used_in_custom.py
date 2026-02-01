#!/usr/bin/env python3
"""
Identify which standard Odoo models are referenced in custom code.

This script compares normalized custom addon reports with normalized standard
Odoo reports to find models that exist in both, indicating custom code uses
standard models.

ARCHITECTURAL NOTES:
====================

This script detects DECLARED touchpoints only:
- Models with exact name matches (custom model extends standard)
- Models inherited via _inherit attribute

This script does NOT detect:
- Runtime model usage: self.env['res.partner'].search(...)
- Dynamic model references
- Indirect model usage

Runtime usage detection is deferred to usage_report.py which uses AST analysis.

This is intentional - this script answers: "Which standard models are declared
or extended in custom code?" not "Which standard models are used at runtime?"
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelMatch:
    """Information about a model match between custom and standard."""
    in_custom: bool
    in_standard: bool
    custom_module: str
    standard_module: str = ""


def find_standard_models_used_in_custom(
    normalized_custom: Dict[str, Dict[str, Any]],
    normalized_standard: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Find models that exist in both custom and standard reports.
    
    Args:
        normalized_custom: Normalized custom addons report
        normalized_standard: Normalized standard Odoo report
        
    Returns:
        Dictionary mapping model_name -> ModelMatch info
    """
    matches = {}
    
    # Get all model names from both reports
    custom_models = set(normalized_custom.keys())
    standard_models = set(normalized_standard.keys())
    
    # Find intersection
    common_models = custom_models & standard_models
    
    # Also check for models that inherit from standard models
    for custom_model_name, custom_data in normalized_custom.items():
        # Check if model itself is in standard
        if custom_model_name in standard_models:
            matches[custom_model_name] = {
                "in_custom": True,
                "in_standard": True,
                "custom_module": custom_data.get('module', 'unknown'),
                "standard_module": normalized_standard[custom_model_name].get('module', 'unknown'),
                "match_type": "exact_name"
            }
        else:
            # Check if model inherits from a standard model
            inherit_list = custom_data.get('inherit', [])
            for inherited_model in inherit_list:
                if inherited_model in standard_models:
                    # Custom model extends standard model
                    if inherited_model not in matches:
                        matches[inherited_model] = {
                            "in_custom": False,
                            "in_standard": True,
                            "custom_module": custom_data.get('module', 'unknown'),
                            "standard_module": normalized_standard[inherited_model].get('module', 'unknown'),
                            "match_type": "inherited_by",
                            "inherited_by": [custom_model_name]
                        }
                    else:
                        # Add to inherited_by list
                        if 'inherited_by' not in matches[inherited_model]:
                            matches[inherited_model]['inherited_by'] = []
                        matches[inherited_model]['inherited_by'].append(custom_model_name)
                    break
    
    # Sort for deterministic output
    return dict(sorted(matches.items()))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Identify standard Odoo models used in custom code"
    )
    parser.add_argument(
        'normalized_custom',
        type=str,
        help='Path to normalized custom addons JSON report'
    )
    parser.add_argument(
        'normalized_standard',
        type=str,
        help='Path to normalized standard Odoo JSON report'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output JSON file (default: standard_models_used.json)'
    )
    
    args = parser.parse_args()
    
    # Load normalized reports
    custom_path = Path(args.normalized_custom)
    standard_path = Path(args.normalized_standard)
    
    if not custom_path.exists():
        print(f"Error: Custom report not found: {custom_path}", file=sys.stderr)
        sys.exit(1)
    
    if not standard_path.exists():
        print(f"Error: Standard report not found: {standard_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading normalized custom report from {custom_path}...")
    with open(custom_path, 'r', encoding='utf-8') as f:
        custom_data = json.load(f)
    
    print(f"Loading normalized standard report from {standard_path}...")
    with open(standard_path, 'r', encoding='utf-8') as f:
        standard_data = json.load(f)
    
    custom_models = custom_data.get('normalized_models', {})
    standard_models = standard_data.get('normalized_models', {})
    
    print(f"Custom models: {len(custom_models)}")
    print(f"Standard models: {len(standard_models)}")
    
    # Find matches
    print("Finding standard models used in custom code...")
    matches = find_standard_models_used_in_custom(custom_models, standard_models)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('addons_reports') / 'standard_models_used.json'
        output_path.parent.mkdir(exist_ok=True)
    
    # Prepare output
    output_data = {
        "source_custom": str(custom_path),
        "source_standard": str(standard_path),
        "total_matches": len(matches),
        "matches": matches
    }
    
    print(f"Found {len(matches)} standard models used in custom code")
    print(f"Writing results to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()

