#!/usr/bin/env python3
"""
Detect actual usage of removed/deprecated APIs in custom code.

This script uses AST analysis to find usage of:
- Removed fields
- Removed methods
- Deprecated BaseModel APIs
- Removed field types
"""

import ast
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class UsageFinding:
    """Represents a single usage finding."""
    category: str  # removed_field, removed_method, deprecated_api, core_api
    model: str
    api_name: str
    file_path: str
    line_number: int
    context: str  # Code snippet


class ModelTracker(ast.NodeVisitor):
    """Track model names from class definitions."""
    
    def __init__(self):
        self.class_to_model: Dict[str, str] = {}  # class_name -> model_name
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class and extract model name."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Check if this is an Odoo model
        is_odoo_model = False
        for base in node.bases:
            if isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name) and base.value.id == 'models':
                    if base.attr == 'Model':
                        is_odoo_model = True
                        break
        
        if is_odoo_model:
            # Extract _name from class body
            model_name = None
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name) and target.id == '_name':
                        if isinstance(stmt.value, ast.Constant):
                            model_name = stmt.value.value
                        elif isinstance(stmt.value, ast.Str):
                            model_name = stmt.value.s
                        break
            
            # Also check _inherit for Extension models
            if not model_name:
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        target = stmt.targets[0]
                        if isinstance(target, ast.Name) and target.id == '_inherit':
                            if isinstance(stmt.value, ast.List):
                                if stmt.value.elts:
                                    first_inherit = stmt.value.elts[0]
                                    if isinstance(first_inherit, ast.Constant):
                                        model_name = first_inherit.value
                                    elif isinstance(first_inherit, ast.Str):
                                        model_name = first_inherit.s
                                    break
            
            if model_name:
                self.class_to_model[node.name] = model_name
            else:
                # Fallback to class name
                self.class_to_model[node.name] = node.name
        
        self.generic_visit(node)
        self.current_class = old_class


class APIUsageDetector(ast.NodeVisitor):
    """Detect API usage in custom code."""
    
    def __init__(
        self,
        filename: str,
        model_gaps: Dict[str, Dict[str, Any]],
        core_api_gaps: Dict[str, Any],
        removed_field_types: List[str],
        class_to_model: Dict[str, str]
    ):
        self.filename = filename
        self.model_gaps = model_gaps
        self.core_api_gaps = core_api_gaps
        self.removed_field_types = set(removed_field_types)
        self.class_to_model = class_to_model
        self.findings: List[UsageFinding] = []
        self.current_class = None
        self.current_method = None
        
        # Build lookup structures
        self.removed_methods = self._extract_removed_methods()
        self.removed_fields = self._extract_removed_fields()
        self.deprecated_methods = {m['name'] for m in core_api_gaps.get('deprecated_methods', [])}
        self.removed_base_methods = {m['name'] for m in core_api_gaps.get('removed_methods', [])}
        
        # Build reverse lookup: field/method -> list of models that have it removed
        self.field_to_models = defaultdict(set)
        self.method_to_models = defaultdict(set)
        for model_name, gap_data in model_gaps.items():
            for field in gap_data.get('removed_fields', []):
                self.field_to_models[field].add(model_name)
            for method in gap_data.get('removed_methods', []):
                self.method_to_models[method].add(model_name)
        
    def _extract_removed_methods(self) -> Dict[str, Set[str]]:
        """Extract removed methods by model."""
        removed = defaultdict(set)
        for model_name, gap_data in self.model_gaps.items():
            for method in gap_data.get('removed_methods', []):
                removed[model_name].add(method)
        return dict(removed)
    
    def _extract_removed_fields(self) -> Dict[str, Set[str]]:
        """Extract removed fields by model."""
        removed = defaultdict(set)
        for model_name, gap_data in self.model_gaps.items():
            for field in gap_data.get('removed_fields', []):
                removed[model_name].add(field)
        return dict(removed)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track current class."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track current method and check decorators."""
        old_method = self.current_method
        self.current_method = node.name
        
        # Check if this method overrides a deprecated/removed BaseModel method
        method_name = node.name
        model_name = self._get_current_model_name()
        
        # Check if this is a deprecated BaseModel method being overridden
        if method_name in self.deprecated_methods:
            context = self._get_context_snippet(node.lineno)
            self.findings.append(UsageFinding(
                category="deprecated_api",
                model=model_name,
                api_name=method_name,
                file_path=self.filename,
                line_number=node.lineno,
                context=context
            ))
        
        # Check if this is a removed BaseModel method being overridden
        if method_name in self.removed_base_methods:
            context = self._get_context_snippet(node.lineno)
            self.findings.append(UsageFinding(
                category="core_api",
                model=model_name,
                api_name=method_name,
                file_path=self.filename,
                line_number=node.lineno,
                context=context
            ))
        
        # Check decorators for deprecated APIs
        for decorator in node.decorator_list:
            self._check_decorator(decorator, node.lineno)
        
        self.generic_visit(node)
        self.current_method = old_method
    
    def _check_decorator(self, decorator: ast.expr, line: int):
        """Check if decorator uses deprecated API."""
        try:
            decorator_str = self._get_decorator_string(decorator)
            if decorator_str:
                # Check for deprecated decorators
                for deprecated in self.deprecated_methods:
                    if deprecated in decorator_str:
                        model_name = self._get_current_model_name()
                        self.findings.append(UsageFinding(
                            category="deprecated_api",
                            model=model_name,
                            api_name=deprecated,
                            file_path=self.filename,
                            line_number=line,
                            context=f"@decorator: {decorator_str}"
                        ))
        except:
            pass
    
    def visit_Attribute(self, node: ast.Attribute):
        """Check attribute access (field access)."""
        attr_name = node.attr
        
        # Check if this field is removed from any model
        if attr_name in self.field_to_models:
            # Try to determine which model this is
            model_name = self._infer_model_from_context(node)
            if model_name:
                # Check if this specific model has the field removed
                if model_name in self.field_to_models[attr_name]:
                    context = self._get_context_snippet(node.lineno)
                    self.findings.append(UsageFinding(
                        category="removed_field",
                        model=model_name,
                        api_name=attr_name,
                        file_path=self.filename,
                        line_number=node.lineno,
                        context=context
                    ))
            else:
                # Field is removed from some model, but we can't determine which
                # Report for all models that have it removed
                for model in self.field_to_models[attr_name]:
                    context = self._get_context_snippet(node.lineno)
                    self.findings.append(UsageFinding(
                        category="removed_field",
                        model=model,
                        api_name=attr_name,
                        file_path=self.filename,
                        line_number=node.lineno,
                        context=context
                    ))
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Check method calls."""
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            # Check for removed methods from model gaps
            if method_name in self.method_to_models:
                model_name = self._infer_model_from_context(node.func)
                if model_name:
                    if model_name in self.method_to_models[method_name]:
                        context = self._get_context_snippet(node.lineno)
                        self.findings.append(UsageFinding(
                            category="removed_method",
                            model=model_name,
                            api_name=method_name,
                            file_path=self.filename,
                            line_number=node.lineno,
                            context=context
                        ))
                else:
                    # Method is removed from some model, report for all
                    for model in self.method_to_models[method_name]:
                        context = self._get_context_snippet(node.lineno)
                        self.findings.append(UsageFinding(
                            category="removed_method",
                            model=model,
                            api_name=method_name,
                            file_path=self.filename,
                            line_number=node.lineno,
                            context=context
                        ))
            
            # Check for deprecated BaseModel methods
            if method_name in self.deprecated_methods:
                model_name = self._get_current_model_name()
                context = self._get_context_snippet(node.lineno)
                self.findings.append(UsageFinding(
                    category="deprecated_api",
                    model=model_name,
                    api_name=method_name,
                    file_path=self.filename,
                    line_number=node.lineno,
                    context=context
                ))
            
            # Check for removed BaseModel methods
            if method_name in self.removed_base_methods:
                model_name = self._get_current_model_name()
                context = self._get_context_snippet(node.lineno)
                self.findings.append(UsageFinding(
                    category="core_api",
                    model=model_name,
                    api_name=method_name,
                    file_path=self.filename,
                    line_number=node.lineno,
                    context=context
                ))
        
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant):
        """Check string constants for domain usage."""
        if isinstance(node.value, str):
            self._check_domain_string(node.value, node.lineno)
        self.generic_visit(node)
    
    def visit_Str(self, node: ast.Str):
        """Check string literals for domain usage (Python < 3.8)."""
        self._check_domain_string(node.s, node.lineno)
        self.generic_visit(node)
    
    def _check_domain_string(self, value: str, line: int):
        """Check if string contains domain with removed fields."""
        # Domain format: [('field_name', 'operator', value), ...]
        # Look for field names in domain strings
        for field_name, models in self.field_to_models.items():
            # Check if field name appears in domain string
            pattern = rf"['\"]{re.escape(field_name)}['\"]"
            if re.search(pattern, value):
                # Report for all models that have this field removed
                for model_name in models:
                    self.findings.append(UsageFinding(
                        category="removed_field",
                        model=model_name,
                        api_name=field_name,
                        file_path=self.filename,
                        line_number=line,
                        context=f"domain: {value[:100]}"
                    ))
    
    def _infer_model_from_context(self, node: ast.expr) -> Optional[str]:
        """Try to infer model name from AST context."""
        # If accessing self.attr, use current class's model name
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id == 'self':
                    return self._get_current_model_name()
                # Check for common recordset variable names
                elif node.value.id in ('record', 'records', 'rec', 'obj'):
                    return self._get_current_model_name()
        
        return None
    
    def _get_current_model_name(self) -> str:
        """Get model name for current class."""
        if self.current_class:
            return self.class_to_model.get(self.current_class, self.current_class)
        return "unknown"
    
    def _get_decorator_string(self, decorator: ast.expr) -> Optional[str]:
        """Convert decorator AST to string."""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                return f"{self._get_decorator_string(decorator.value)}.{decorator.attr}"
            elif isinstance(decorator, ast.Call):
                func = self._get_decorator_string(decorator.func)
                return func
        except:
            pass
        return None
    
    def _get_context_snippet(self, line: int, context_lines: int = 2) -> str:
        """Get code snippet around a line number."""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start = max(0, line - context_lines - 1)
            end = min(len(lines), line + context_lines)
            
            snippet_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line - 1 else "    "
                snippet_lines.append(f"{prefix}{lines[i].rstrip()}")
            
            return "\n".join(snippet_lines)
        except:
            return f"Line {line}"


def analyze_file(
    file_path: Path,
    model_gaps: Dict[str, Dict[str, Any]],
    core_api_gaps: Dict[str, Any]
) -> List[UsageFinding]:
    """Analyze a Python file for API usage."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(file_path))
        
        # First pass: track model names
        tracker = ModelTracker()
        tracker.visit(tree)
        
        # Second pass: detect usage
        removed_field_types = [ft['name'] for ft in core_api_gaps.get('removed_field_types', [])]
        
        detector = APIUsageDetector(
            str(file_path),
            model_gaps,
            core_api_gaps,
            removed_field_types,
            tracker.class_to_model
        )
        detector.visit(tree)
        
        return detector.findings
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Warning: Error analyzing {file_path}: {e}", file=sys.stderr)
        return []


def generate_aggregate_report(findings: List[UsageFinding]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Generate aggregate report grouped by model and API."""
    aggregate = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for finding in findings:
        if finding.category == "removed_field":
            aggregate[finding.model]["removed_fields"][finding.api_name] += 1
        elif finding.category == "removed_method":
            aggregate[finding.model]["removed_methods"][finding.api_name] += 1
        elif finding.category == "deprecated_api":
            aggregate[finding.model]["deprecated_apis"][finding.api_name] += 1
        elif finding.category == "core_api":
            aggregate[finding.model]["core_apis"][finding.api_name] += 1
    
    # Convert to regular dict and sort
    result = {}
    for model in sorted(aggregate.keys()):
        result[model] = {}
        for category in sorted(aggregate[model].keys()):
            result[model][category] = dict(sorted(aggregate[model][category].items()))
    
    return result


def extract_module_name(file_path: str) -> str:
    """Extract module name from file path."""
    # Path format: addons_folder/module_name/models/file.py
    # or: addons_folder/module_name/wizard/file.py
    parts = Path(file_path).parts
    try:
        # Find 'addons_folder' or similar and get next part
        for i, part in enumerate(parts):
            if 'addon' in part.lower() or part in ['fsf_base', 'fsf_setting']:
                if i + 1 < len(parts):
                    return parts[i + 1]
        # Fallback: use first directory after addons_folder
        if len(parts) > 1:
            return parts[1]
    except:
        pass
    return "unknown"


def sanitize_for_path(name: str) -> str:
    """Convert model/module name to safe filesystem path."""
    # Replace dots, spaces, and special chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized if sanitized else 'unknown'


def generate_category_filename(category_key: str) -> str:
    """Convert category key to filename."""
    mapping = {
        "removed_field": "removed_fields.md",
        "removed_method": "removed_methods.md",
        "deprecated_api": "deprecated_apis.md",
        "core_api": "core_api_changes.md"
    }
    return mapping.get(category_key, f"{category_key}.md")


def generate_markdown_report(
    findings: List[UsageFinding],
    aggregate: Dict[str, Dict[str, Dict[str, int]]],
    output_dir: Path
) -> None:
    """
    Generate structured markdown reports in folder hierarchy.
    
    Structure:
    migration_report/
      index.md (summary)
      module_name/
        index.md (module summary)
        model_name/
          index.md (model summary)
          removed_fields.md
          removed_methods.md
          deprecated_apis.md
          core_api_changes.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_findings = len(findings)
    
    # Generate root index.md
    root_index = []
    root_index.append("# Code Migration Report")
    root_index.append("")
    root_index.append("This report identifies code that needs to be migrated for Odoo 14 → 19.")
    root_index.append("")
    root_index.append("## Summary")
    root_index.append("")
    root_index.append(f"- **Total findings**: {total_findings}")
    root_index.append(f"- **Models affected**: {len(aggregate)}")
    root_index.append("")
    
    if total_findings == 0:
        root_index.append("No migration issues found!")
        with open(output_dir / "index.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(root_index))
        return
    
    # Organize by: module -> model -> category -> API -> findings
    by_module = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for finding in findings:
        module_name = extract_module_name(finding.file_path)
        by_module[module_name][finding.model][finding.category][finding.api_name].append(finding)
    
    # Category display names
    category_names = {
        "removed_field": "Removed Fields",
        "removed_method": "Removed Methods",
        "deprecated_api": "Deprecated APIs",
        "core_api": "Core API Changes"
    }
    
    # Generate module links for root index
    root_index.append("## Modules")
    root_index.append("")
    for module_name in sorted(by_module.keys()):
        module_safe = sanitize_for_path(module_name)
        root_index.append(f"- [{module_name}]({module_safe}/index.md)")
    root_index.append("")
    
    with open(output_dir / "index.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(root_index))
    
    # Generate reports for each module
    for module_name in sorted(by_module.keys()):
        module_safe = sanitize_for_path(module_name)
        module_dir = output_dir / module_safe
        module_dir.mkdir(exist_ok=True)
        
        module_data = by_module[module_name]
        
        # Generate module index
        module_index = []
        module_index.append(f"# Module: {module_name}")
        module_index.append("")
        total_module_findings = sum(len(api_data) for model_data in module_data.values() for cat_data in model_data.values() for api_data in cat_data.values())
        module_index.append(f"**Total findings**: {total_module_findings}")
        module_index.append(f"**Models affected**: {len(module_data)}")
        module_index.append("")
        module_index.append("## Models")
        module_index.append("")
        
        for model_name in sorted(module_data.keys()):
            model_safe = sanitize_for_path(model_name)
            module_index.append(f"- [{model_name}]({model_safe}/index.md)")
        module_index.append("")
        
        with open(module_dir / "index.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(module_index))
        
        # Generate reports for each model
        for model_name in sorted(module_data.keys()):
            model_safe = sanitize_for_path(model_name)
            model_dir = module_dir / model_safe
            model_dir.mkdir(exist_ok=True)
            
            model_data = module_data[model_name]
            
            # Generate model index
            model_index = []
            model_index.append(f"# Model: {model_name}")
            model_index.append("")
            model_index.append(f"**Module**: {module_name}")
            model_index.append("")
            
            total_model_findings = sum(len(api_data) for cat_data in model_data.values() for api_data in cat_data.values())
            model_index.append(f"**Total findings**: {total_model_findings}")
            model_index.append("")
            model_index.append("## Categories")
            model_index.append("")
            
            for category_key in sorted(model_data.keys()):
                category_name = category_names.get(category_key, category_key)
                category_file = generate_category_filename(category_key)
                category_findings_count = sum(len(api_data) for api_data in model_data[category_key].values())
                model_index.append(f"- [{category_name}]({category_file}) ({category_findings_count} APIs)")
            model_index.append("")
            
            with open(model_dir / "index.md", 'w', encoding='utf-8') as f:
                f.write("\n".join(model_index))
            
            # Generate category files
            for category_key in sorted(model_data.keys()):
                category_name = category_names.get(category_key, category_key)
                category_file = model_dir / generate_category_filename(category_key)
                
                category_lines = []
                category_lines.append(f"# {category_name}")
                category_lines.append("")
                category_lines.append(f"**Model**: {model_name}")
                category_lines.append(f"**Module**: {module_name}")
                category_lines.append("")
                
                category_data = model_data[category_key]
                
                for api_name in sorted(category_data.keys()):
                    api_findings = category_data[api_name]
                    count = len(api_findings)
                    category_lines.append(f"## {api_name} ({count} occurrence{'s' if count > 1 else ''})")
                    category_lines.append("")
                    
                    # Show occurrences grouped by file
                    by_file = defaultdict(list)
                    for finding in api_findings:
                        by_file[finding.file_path].append(finding)
                    
                    for file_path in sorted(by_file.keys()):
                        file_findings = by_file[file_path]
                        category_lines.append(f"### File: `{file_path}`")
                        category_lines.append("")
                        
                        for finding in file_findings:
                            category_lines.append(f"**Line {finding.line_number}:**")
                            category_lines.append("")
                            category_lines.append("```python")
                            category_lines.append(finding.context)
                            category_lines.append("```")
                            category_lines.append("")
                    
                    category_lines.append("")
                
                with open(category_file, 'w', encoding='utf-8') as f:
                    f.write("\n".join(category_lines))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect usage of removed/deprecated APIs in custom code"
    )
    parser.add_argument(
        'custom_addons_path',
        type=str,
        help='Path to custom addons folder'
    )
    parser.add_argument(
        'model_gaps',
        type=str,
        help='Path to JSON output from standv14_used_in_custom_no_exist_v19.py'
    )
    parser.add_argument(
        'core_api_gaps',
        type=str,
        help='Path to JSON output from core_api_gap_report.py'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output JSON file (default: usage_report.json)'
    )
    parser.add_argument(
        '--markdown',
        type=str,
        help='Path to output Markdown file (optional)'
    )
    
    args = parser.parse_args()
    
    # Load gap reports
    gaps_path = Path(args.model_gaps)
    core_path = Path(args.core_api_gaps)
    
    if not gaps_path.exists():
        print(f"Error: Model gaps file not found: {gaps_path}", file=sys.stderr)
        sys.exit(1)
    
    if not core_path.exists():
        print(f"Error: Core API gaps file not found: {core_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading model gaps from {gaps_path}...")
    with open(gaps_path, 'r', encoding='utf-8') as f:
        gaps_data = json.load(f)
    
    print(f"Loading core API gaps from {core_path}...")
    with open(core_path, 'r', encoding='utf-8') as f:
        core_data = json.load(f)
    
    model_gaps = gaps_data.get('gaps', {})
    print(f"Loaded {len(model_gaps)} models with gaps")
    print(f"Deprecated methods: {len(core_data.get('deprecated_methods', []))}")
    print(f"Removed base methods: {len(core_data.get('removed_methods', []))}")
    
    # Find Python files in custom addons
    addons_path = Path(args.custom_addons_path)
    if not addons_path.exists():
        print(f"Error: Custom addons path not found: {addons_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning {addons_path} for Python files...")
    python_files = list(addons_path.rglob('*.py'))
    # Exclude __pycache__ and test files if needed
    python_files = [f for f in python_files if '__pycache__' not in str(f)]
    print(f"Found {len(python_files)} Python files")
    
    # Analyze files
    all_findings = []
    for py_file in python_files:
        findings = analyze_file(py_file, model_gaps, core_data)
        all_findings.extend(findings)
        if findings:
            print(f"Found {len(findings)} issues in {py_file}")
    
    print(f"\nTotal findings: {len(all_findings)}")
    
    # Generate aggregate report
    aggregate = generate_aggregate_report(all_findings)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('addons_reports') / 'usage_report.json'
        output_path.parent.mkdir(exist_ok=True)
    
    # Prepare JSON output
    output_data = {
        "source_addons": str(addons_path),
        "source_model_gaps": str(gaps_path),
        "source_core_api_gaps": str(core_path),
        "total_findings": len(all_findings),
        "aggregate": aggregate,
        "detailed": [
            {
                "category": f.category,
                "model": f.model,
                "api_name": f.api_name,
                "file_path": f.file_path,
                "line_number": f.line_number,
                "context": f.context
            }
            for f in all_findings
        ]
    }
    
    print(f"Writing JSON report to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON output saved to: {output_path}")
    
    # Generate markdown if requested
    if args.markdown:
        md_path = Path(args.markdown)
        print(f"Generating structured Markdown reports...")
        
        # If path ends with .md, use parent directory; otherwise use as directory
        if md_path.suffix == '.md':
            output_dir = md_path.parent / md_path.stem
        else:
            output_dir = md_path
        
        generate_markdown_report(all_findings, aggregate, output_dir)
        
        print(f"Markdown reports saved to: {output_dir}/")
        print(f"  - Root index: {output_dir}/index.md")


if __name__ == '__main__':
    main()
