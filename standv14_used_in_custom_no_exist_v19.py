#!/usr/bin/env python3
"""
Identify APIs removed between Odoo 14 and 19 for models used in custom code.

This script compares the Odoo 14 API surface (from models used in custom code)
with the Odoo 19 equivalent to identify removed fields and methods.

ARCHITECTURAL NOTE:
===================

This script analyzes MODEL-LEVEL APIs only (declared methods on models).
BaseModel methods (create, write, unlink, search, etc.) are NOT declared on
individual models - they are inherited from BaseModel.

BaseModel API changes are handled exclusively by core_api_gap_report.py.

CRITICAL: BaseModel methods must NEVER appear in model-level gap reports.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

# BaseModel methods that are inherited, not declared on individual models
# These MUST be filtered out from model-level gap analysis
BASEMODEL_METHODS = {
    "create",
    "write",
    "unlink",
    "read",
    "search",
    "search_read",
    "browse",
    "copy",
    "exists",
    "ensure_one",
    "mapped",
    "filtered",
    "sudo",
    "with_context",
    "with_company",
    "with_env",
    "with_user",
    "with_uid",
    "with_prefetch",
    "env",
    "_env",
}


def find_api_gaps(
    s14_api_surface: Dict[str, Dict[str, Any]],
    normalized_s19: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Find API gaps between Odoo 14 and 19 for models used in custom code.
    
    Args:
        s14_api_surface: API surface from standv14_used_in_custom.py
        normalized_s19: Normalized Odoo 19 standard report
        
    Returns:
        Dictionary mapping model_name -> gap information
    """
    gaps = {}
    
    s19_models = normalized_s19.get('normalized_models', {})
    api_surface = s14_api_surface.get('api_surface', {})
    
    for model_name, s14_data in api_surface.items():
        # Use declared_fields and declared_methods (lossless declarations only)
        s14_fields = set(s14_data.get('declared_fields', []))
        s14_methods = set(s14_data.get('declared_methods', []))
        
        # CRITICAL: Filter out BaseModel methods - these are NOT model-level APIs
        # They are inherited from BaseModel and handled by core_api_gap_report.py
        s14_methods = {m for m in s14_methods if m not in BASEMODEL_METHODS}
        
        removed_fields = []
        removed_methods = []
        
        if model_name not in s19_models:
            # Model doesn't exist in Odoo 19 - all declared fields/methods are removed
            removed_fields = sorted(list(s14_fields))
            removed_methods = sorted(list(s14_methods))
        else:
            # Model exists - compare declared fields and methods
            s19_data = s19_models[model_name]
            s19_fields = set(s19_data.get('declared_fields', []))
            s19_methods = set(s19_data.get('declared_methods', []))
            
            # CRITICAL: Also filter BaseModel methods from v19 comparison
            s19_methods = {m for m in s19_methods if m not in BASEMODEL_METHODS}
            
            # Find removed fields (declared in v14 but not in v19)
            removed_fields = sorted(list(s14_fields - s19_fields))
            
            # Find removed methods (declared in v14 but not in v19)
            # Note: BaseModel methods are already filtered out above
            removed_methods = sorted(list(s14_methods - s19_methods))
        
        # Only add to gaps if there are actual removals
        if removed_fields or removed_methods:
            gaps[model_name] = {
                "removed_fields": removed_fields,
                "removed_methods": removed_methods,
                "module": s14_data.get('module', 'unknown'),
                "model_type": s14_data.get('model_type', 'unknown'),
                "model_exists_in_v19": model_name in s19_models
            }
    
    # Sort for deterministic output
    return dict(sorted(gaps.items()))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Identify API gaps between Odoo 14 and 19"
    )
    parser.add_argument(
        's14_api_surface',
        type=str,
        help='Path to JSON output from standv14_used_in_custom.py'
    )
    parser.add_argument(
        'normalized_s19',
        type=str,
        help='Path to normalized Odoo 19 standard JSON report'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output JSON file (default: standv14_no_exist_v19.json)'
    )
    
    args = parser.parse_args()
    
    # Load input files
    s14_path = Path(args.s14_api_surface)
    s19_path = Path(args.normalized_s19)
    
    if not s14_path.exists():
        print(f"Error: S14 API surface file not found: {s14_path}", file=sys.stderr)
        sys.exit(1)
    
    if not s19_path.exists():
        print(f"Error: Normalized S19 report not found: {s19_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading S14 API surface from {s14_path}...")
    with open(s14_path, 'r', encoding='utf-8') as f:
        s14_data = json.load(f)
    
    print(f"Loading normalized Odoo 19 report from {s19_path}...")
    with open(s19_path, 'r', encoding='utf-8') as f:
        s19_data = json.load(f)
    
    # Find API gaps
    print("Finding API gaps between Odoo 14 and 19...")
    gaps = find_api_gaps(s14_data, s19_data)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('addons_reports') / 'standv14_no_exist_v19.json'
        output_path.parent.mkdir(exist_ok=True)
    
    # Calculate statistics
    total_removed_fields = sum(len(m.get('removed_fields', [])) for m in gaps.values())
    total_removed_methods = sum(len(m.get('removed_methods', [])) for m in gaps.values())
    models_with_gaps = len(gaps)
    
    # Prepare output
    output_data = {
        "source_s14_api_surface": str(s14_path),
        "source_normalized_s19": str(s19_path),
        "total_models_with_gaps": models_with_gaps,
        "total_removed_fields": total_removed_fields,
        "total_removed_methods": total_removed_methods,
        "gaps": gaps
    }
    
    print(f"Found gaps in {models_with_gaps} models")
    print(f"Total removed fields: {total_removed_fields}")
    print(f"Total removed methods: {total_removed_methods}")
    print(f"Writing results to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()

