#!/usr/bin/env python3
"""
Extract BaseModel and core API changes from Odoo source code.

This script analyzes orm_14 and orm_19 source files to identify:
- Removed/deprecated BaseModel methods
- Removed/deprecated decorators
- Removed/deprecated field types

ARCHITECTURAL NOTES:
===================

This script is the ONLY one that analyzes BaseModel/ORM APIs.
Model-level APIs are handled by normalize_reports.py and downstream scripts.

Classification System:
- "removed": Method exists in v14 but NOT in v19 (truly removed)
- "deprecated": Method exists in both v14 and v19, marked deprecated in v19
- "migrated": Method exists in both, deprecated in v19 with replacement hint

Important: Methods that exist but are deprecated are NOT "removed".
This prevents false positives for methods like check_access_rights which
still exist in v19 but are deprecated in favor of check_access().

This script uses AST parsing, not heuristics, to ensure accuracy.
"""

import ast
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class APIMethod:
    """Represents a method in BaseModel or related classes."""
    name: str
    file: str
    line: int
    deprecated: bool
    replacement: Optional[str] = None
    decorators: List[str] = None
    
    def __post_init__(self):
        if self.decorators is None:
            self.decorators = []


@dataclass
class FieldType:
    """Represents a field type class."""
    name: str
    file: str
    line: int
    deprecated: bool
    replacement: Optional[str] = None


class BaseModelMethodExtractor(ast.NodeVisitor):
    """Extract methods from BaseModel and related classes."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.methods: List[APIMethod] = []
        self.current_class = None
        self.base_model_classes = {'BaseModel', 'Model', 'TransientModel', 'AbstractModel'}
        self.in_base_model = False
        self.processed_classes = {}  # Track which classes are BaseModel classes
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions."""
        old_class = self.current_class
        old_in_base = self.in_base_model
        
        self.current_class = node.name
        
        # Check if this is BaseModel or inherits from BaseModel
        # In Odoo 19, BaseModel uses metaclass: class BaseModel(metaclass=MetaModel)
        # AbstractModel = BaseModel (assignment, not inheritance)
        # Model(AbstractModel) inherits from AbstractModel
        
        # Check if class name itself is a BaseModel class
        if node.name in self.base_model_classes:
            self.in_base_model = True
            self.processed_classes[node.name] = True
        else:
            # Check if it inherits from a known BaseModel class
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                
                if base_name:
                    # Check if base is a known BaseModel class
                    if base_name in self.base_model_classes or self.processed_classes.get(base_name):
                        self.in_base_model = True
                        self.processed_classes[node.name] = True
                        break
        
        self.generic_visit(node)
        
        self.current_class = old_class
        self.in_base_model = old_in_base
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions."""
        if self.in_base_model:
            # Include all methods (public and private) as deprecated methods can be either
            # Skip only truly internal Python methods (like __init__, __str__, etc.)
            # but include custom private methods (like _filter_access_rules)
            if not node.name.startswith('__'):
                deprecated = False
                replacement = None
                decorators = []
                
                # Check decorators for deprecation
                for decorator in node.decorator_list:
                    decorator_str = self._get_decorator_string(decorator)
                    if decorator_str:
                        decorators.append(decorator_str)
                        if 'deprecated' in decorator_str.lower():
                            deprecated = True
                            # Try to extract replacement from decorator
                            replacement = self._extract_replacement(decorator)
                
                # Check docstring for deprecation
                if ast.get_docstring(node):
                    docstring = ast.get_docstring(node).lower()
                    if 'deprecated' in docstring:
                        deprecated = True
                
                method = APIMethod(
                    name=node.name,
                    file=self.filename,
                    line=node.lineno,
                    deprecated=deprecated,
                    replacement=replacement,
                    decorators=decorators
                )
                self.methods.append(method)
        
        self.generic_visit(node)
    
    def _get_decorator_string(self, decorator: ast.expr) -> Optional[str]:
        """Convert decorator AST node to string."""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                return f"{self._get_decorator_string(decorator.value)}.{decorator.attr}"
            elif isinstance(decorator, ast.Call):
                func = self._get_decorator_string(decorator.func)
                args = ', '.join(ast.unparse(arg) if hasattr(ast, 'unparse') else str(arg) for arg in decorator.args)
                return f"{func}({args})"
        except:
            pass
        return None
    
    def _extract_replacement(self, decorator: ast.expr) -> Optional[str]:
        """Extract replacement method name from deprecated decorator."""
        try:
            if isinstance(decorator, ast.Call):
                for keyword in decorator.keywords:
                    if keyword.arg == 'replacement' or 'replacement' in str(keyword.value).lower():
                        if isinstance(keyword.value, ast.Constant):
                            return keyword.value.value
                        elif isinstance(keyword.value, ast.Str):
                            return keyword.value.s
                # Check args for replacement info
                for arg in decorator.args:
                    arg_str = ast.unparse(arg) if hasattr(ast, 'unparse') else str(arg)
                    # Look for patterns like "use X() instead" or "use X instead" or "replaced by X"
                    # Match method calls like "use check_access()" or just "use check_access"
                    match = re.search(r'(?:use|replaced by|replacement:)\s+([a-z_][a-z0-9_]*)\s*(?:\(\))?', arg_str, re.I)
                    if match:
                        return match.group(1)
                    # Also check for patterns like "use `method_name`"
                    match = re.search(r'use\s+`([a-z_][a-z0-9_]*)`', arg_str, re.I)
                    if match:
                        return match.group(1)
        except:
            pass
        return None


class FieldTypeExtractor(ast.NodeVisitor):
    """Extract field type classes from fields.py."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.field_types: List[FieldType] = []
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions to find field types."""
        # Field types typically inherit from Field or have Field in name
        is_field_type = False
        
        for base in node.bases:
            base_str = self._get_base_string(base)
            if 'Field' in base_str or node.name.endswith('Field'):
                is_field_type = False
                # Check if it's actually a field type (not Field itself)
                if node.name != 'Field' and not node.name.startswith('_'):
                    is_field_type = True
                    break
        
        if is_field_type:
            deprecated = False
            replacement = None
            
            # Check docstring for deprecation
            if ast.get_docstring(node):
                docstring = ast.get_docstring(node).lower()
                if 'deprecated' in docstring:
                    deprecated = True
            
            field_type = FieldType(
                name=node.name,
                file=self.filename,
                line=node.lineno,
                deprecated=deprecated,
                replacement=replacement
            )
            self.field_types.append(field_type)
        
        self.generic_visit(node)
    
    def _get_base_string(self, base: ast.expr) -> str:
        """Convert base class AST to string."""
        try:
            if isinstance(base, ast.Name):
                return base.id
            elif isinstance(base, ast.Attribute):
                return f"{self._get_base_string(base.value)}.{base.attr}"
        except:
            pass
        return ""


def extract_methods_from_file(file_path: Path) -> List[APIMethod]:
    """Extract methods from a models.py file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source, filename=str(file_path))
        extractor = BaseModelMethodExtractor(str(file_path))
        extractor.visit(tree)
        return extractor.methods
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return []


def extract_field_types_from_file(file_path: Path) -> List[FieldType]:
    """Extract field types from a fields.py file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source, filename=str(file_path))
        extractor = FieldTypeExtractor(str(file_path))
        extractor.visit(tree)
        return extractor.field_types
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return []


def extract_decorators_from_api(file_path: Path) -> List[str]:
    """Extract decorator names from api module (if available)."""
    # This is a simplified version - in practice, api decorators are in a separate module
    # For now, we'll return common decorators
    common_decorators = [
        'api.model',
        'api.depends',
        'api.onchange',
        'api.constrains',
        'api.returns',
        'api.multi',
        'api.one',
        'api.v7',
        'api.v8',
        'api.deprecated'
    ]
    return common_decorators


def compare_apis(
    v14_methods: List[APIMethod],
    v19_methods: List[APIMethod],
    v14_field_types: List[FieldType],
    v19_field_types: List[FieldType]
) -> Dict[str, Any]:
    """
    Compare v14 and v19 APIs to find gaps.
    
    This function correctly distinguishes between:
    - removed: Method exists in v14 but NOT in v19 (truly removed)
    - deprecated: Method exists in both v14 and v19, marked deprecated in v19
    - migrated: Method exists in both, deprecated in v19 with replacement hint
    
    CRITICAL: Methods that exist but are deprecated are NOT classified as "removed".
    This prevents false positives. Example: check_access_rights exists in v19 but
    is deprecated in favor of check_access() - it's "migrated", not "removed".
    
    Returns:
        Dictionary with:
        - removed_methods: List of truly removed methods
        - deprecated_methods: List of deprecated methods (includes migrated)
        - removed_field_types: List of removed field types
    """
    # Build lookup dictionaries
    v19_method_dict = {m.name: m for m in v19_methods}
    v19_method_names = set(v19_method_dict.keys())
    v19_field_names = {f.name for f in v19_field_types}
    
    # Find removed methods (exist in v14, not in v19)
    removed_methods = []
    deprecated_methods = []
    
    for method in v14_methods:
        if method.name not in v19_method_names:
            # Truly removed - doesn't exist in v19
            removed_methods.append({
                "name": method.name,
                "file": method.file,
                "line": method.line,
                "deprecated": method.deprecated,
                "replacement": method.replacement,
                "status": "removed"
            })
        else:
            # Method exists in v19 - check if deprecated
            v19_method = v19_method_dict[method.name]
            if v19_method.deprecated:
                # Exists but deprecated - extract replacement if available
                deprecated_methods.append({
                    "name": method.name,
                    "file": method.file,
                    "line": method.line,
                    "deprecated": True,
                    "replacement": v19_method.replacement or method.replacement,
                    "status": "deprecated" if not v19_method.replacement else "migrated",
                    "v19_file": v19_method.file,
                    "v19_line": v19_method.line
                })
    
    # Also find methods that are deprecated in v19 but didn't exist in v14
    # (newly deprecated methods - less common but possible)
    v14_method_names = {m.name for m in v14_methods}
    for method in v19_methods:
        if method.deprecated and method.name not in v14_method_names:
            deprecated_methods.append({
                "name": method.name,
                "file": method.file,
                "line": method.line,
                "deprecated": True,
                "replacement": method.replacement,
                "status": "deprecated",
                "v19_file": method.file,
                "v19_line": method.line
            })
    
    # Find removed field types
    removed_field_types = []
    for field_type in v14_field_types:
        if field_type.name not in v19_field_names:
            removed_field_types.append({
                "name": field_type.name,
                "file": field_type.file,
                "line": field_type.line,
                "deprecated": field_type.deprecated,
                "replacement": field_type.replacement
            })
    
    return {
        "removed_methods": sorted(removed_methods, key=lambda x: x['name']),
        "deprecated_methods": sorted(deprecated_methods, key=lambda x: x['name']),
        "removed_field_types": sorted(removed_field_types, key=lambda x: x['name']),
        "removed_decorators": []  # Would need api module analysis
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract BaseModel and core API changes from Odoo source"
    )
    parser.add_argument(
        'orm_14_dir',
        type=str,
        help='Path to orm_14 directory'
    )
    parser.add_argument(
        'orm_19_dir',
        type=str,
        help='Path to orm_19 directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output JSON file (default: core_api_gap_report.json)'
    )
    
    args = parser.parse_args()
    
    orm_14_path = Path(args.orm_14_dir)
    orm_19_path = Path(args.orm_19_dir)
    
    if not orm_14_path.exists():
        print(f"Error: orm_14 directory not found: {orm_14_path}", file=sys.stderr)
        sys.exit(1)
    
    if not orm_19_path.exists():
        print(f"Error: orm_19 directory not found: {orm_19_path}", file=sys.stderr)
        sys.exit(1)
    
    models_14_path = orm_14_path / 'models.py'
    models_19_path = orm_19_path / 'models.py'
    fields_14_path = orm_14_path / 'fields.py'
    fields_19_path = orm_19_path / 'fields.py'
    
    # Extract methods
    print(f"Extracting methods from {models_14_path}...")
    v14_methods = extract_methods_from_file(models_14_path)
    print(f"Found {len(v14_methods)} methods in Odoo 14")
    
    print(f"Extracting methods from {models_19_path}...")
    v19_methods = extract_methods_from_file(models_19_path)
    print(f"Found {len(v19_methods)} methods in Odoo 19")
    
    # Extract field types
    print(f"Extracting field types from {fields_14_path}...")
    v14_field_types = extract_field_types_from_file(fields_14_path)
    print(f"Found {len(v14_field_types)} field types in Odoo 14")
    
    print(f"Extracting field types from {fields_19_path}...")
    v19_field_types = extract_field_types_from_file(fields_19_path)
    print(f"Found {len(v19_field_types)} field types in Odoo 19")
    
    # Compare APIs
    print("Comparing APIs...")
    gaps = compare_apis(v14_methods, v19_methods, v14_field_types, v19_field_types)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('addons_reports') / 'core_api_gap_report.json'
        output_path.parent.mkdir(exist_ok=True)
    
    # Prepare output
    output_data = {
        "source_orm_14": str(orm_14_path),
        "source_orm_19": str(orm_19_path),
        "total_removed_methods": len(gaps['removed_methods']),
        "total_deprecated_methods": len(gaps['deprecated_methods']),
        "total_removed_field_types": len(gaps['removed_field_types']),
        **gaps
    }
    
    print(f"Removed methods: {len(gaps['removed_methods'])}")
    print(f"Deprecated methods: {len(gaps['deprecated_methods'])}")
    print(f"Removed field types: {len(gaps['removed_field_types'])}")
    print(f"Writing results to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()

