#!/usr/bin/env python3

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
import argparse
from datetime import datetime
import json
import re


class OdooModelAnalyzer(ast.NodeVisitor):
    """Advanced AST visitor for comprehensive Odoo model analysis."""

    def __init__(self):
        self.models: List[Dict[str, Any]] = []
        self.current_class: Optional[Dict[str, Any]] = None
        self.current_module: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Identify Odoo models by checking base classes."""
        if self._is_odoo_model(node.bases):
            self._process_model_class(node)
        else:
            self.generic_visit(node)

    def _is_odoo_model(self, bases: List[ast.expr]) -> bool:
        """Check if class inherits from models.Model."""
        for base in bases:
            if (isinstance(base, ast.Attribute) and
                    isinstance(base.value, ast.Name) and
                    base.value.id == 'models' and
                    base.attr == 'Model'):
                return True
        return False

    def _process_model_class(self, node: ast.ClassDef) -> None:
        """Extract complete model information."""
        self.current_class = {
            'class_name': node.name,
            'model_name': None,
            'model_type': 'unknown',
            'inherit': [],
            'inherits': {},
            'description': self._extract_docstring(node),
            'regular_fields': [],
            'computed_stored': [],
            'computed_non_stored': [],
            'related_fields': [],
            'methods': [],
            'constraints': [],
            'line_number': node.lineno
        }

        self._analyze_class_body(node.body)
        self._classify_model()
        self.models.append(self.current_class.copy())
        self.current_class = None

    def _analyze_class_body(self, body: List[ast.stmt]) -> None:
        """Analyze all class body statements."""
        for stmt in body:
            self._process_statement(stmt)

    def _process_statement(self, stmt: ast.stmt) -> None:
        """Dispatch statement to appropriate handler."""
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and target.id in ('_name', '_inherit', '_inherits', '_description'):
                self._handle_model_attribute(stmt)
            else:
                field_info = self._parse_field(stmt)
                if field_info:
                    category = field_info.pop('category')
                    self.current_class[category].append(field_info)
        elif isinstance(stmt, ast.FunctionDef):
            self._handle_method(stmt)

    def _handle_model_attribute(self, stmt: ast.Assign) -> None:
        """Handle _name, _inherit, _inherits, _description assignments."""
        target_name = stmt.targets[0].id
        value = self._safe_unparse(stmt.value)

        if target_name == '_name':
            self.current_class['model_name'] = value.strip("'\"")
        elif target_name == '_inherit':
            self.current_class['inherit'] = self._parse_inherit_list(value)
        elif target_name == '_inherits':
            self.current_class['inherits'] = self._parse_inherits_dict(value)

    def _parse_field(self, stmt: ast.Assign) -> Optional[Dict]:
        """Parse field declaration: field_name = fields.Type(attr=val, ...)"""
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None

        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            return None

        field_name = target.id

        if not isinstance(stmt.value, ast.Call):
            return None

        call = stmt.value

        if (not isinstance(call.func, ast.Attribute) or
                not isinstance(call.func.value, ast.Name) or
                call.func.value.id != 'fields'):
            return None

        field_type = call.func.attr
        attrs = {}

        for kw in call.keywords:
            if kw.arg:
                attrs[kw.arg] = self._safe_unparse(kw.value)

        field_info = {
            'name': field_name,
            'type': field_type,
            'attrs': attrs,
            'category': 'regular_fields',
            'stored': None
        }

        if 'related' in attrs:
            store_val = attrs.get('store', '')
            if 'True' in str(store_val):
                field_info['category'] = 'related_fields'
                field_info['stored'] = True
            else:
                field_info['category'] = 'related_fields'
                field_info['stored'] = False
        elif 'compute' in attrs:
            store_val = attrs.get('store', '')
            if 'True' in str(store_val):
                field_info['category'] = 'computed_stored'
                field_info['stored'] = True
            else:
                field_info['category'] = 'computed_non_stored'
                field_info['stored'] = False

        return field_info

    def _handle_method(self, node: ast.FunctionDef) -> None:
        """Parse method with comprehensive decorator analysis."""
        method_info = {
            'name': node.name,
            'decorators': self._parse_decorators(node.decorator_list),
            'has_super': self._detect_super_call(node),
            'line_number': node.lineno,
            'docstring': ast.get_docstring(node)
        }
        self.current_class['methods'].append(method_info)

    def _parse_decorators(self, decorator_list: List[ast.expr]) -> List[str]:
        """Parse ALL Odoo API decorator variations."""
        decorators = []
        for decorator in decorator_list:
            decorator_str = self._parse_single_decorator(decorator)
            if decorator_str:
                decorators.append(decorator_str)
        return decorators

    def _parse_single_decorator(self, decorator: ast.expr) -> Optional[str]:
        """Handle all decorator syntax variations."""
        try:
            if isinstance(decorator, ast.Name) and decorator.id == 'api':
                return '@api'

            if isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name):
                if decorator.value.id == 'api':
                    return f"@api.{decorator.attr}"

            if isinstance(decorator, ast.Call):
                return self._parse_decorator_call(decorator)

        except Exception:
            pass
        return None

    def _parse_decorator_call(self, call: ast.Call) -> Optional[str]:
        """Parse @api.decorator(args) syntax."""
        if (isinstance(call.func, ast.Attribute) and
                isinstance(call.func.value, ast.Name) and
                call.func.value.id == 'api'):
            args_str = ', '.join(self._safe_unparse(arg) for arg in call.args)
            return f"@api.{call.func.attr}({args_str})"
        return None

    def _detect_super_call(self, node: ast.FunctionDef) -> bool:
        """Detect super() calls in method body."""

        class SuperDetector(ast.NodeVisitor):
            def __init__(self):
                self.found = False

            def visit_Call(self, call_node):
                if (isinstance(call_node.func, ast.Attribute) and
                        isinstance(call_node.func.value, ast.Call) and
                        isinstance(call_node.func.value.func, ast.Name) and
                        call_node.func.value.func.id == 'super'):
                    self.found = True
                self.generic_visit(call_node)

        detector = SuperDetector()
        detector.visit(node)
        return detector.found

    def _classify_model(self) -> None:
        """Classify model using Odoo official inheritance types."""
        if self.current_class.get('inherits'):
            self.current_class['model_type'] = 'Delegation'
            return

        if not self.current_class['model_name']:
            self.current_class['model_type'] = 'Extension'
            return

        inherit_models = self.current_class['inherit']
        if inherit_models and any(m not in ['mail.thread', 'mail.activity.mixin'] for m in inherit_models):
            self.current_class['model_type'] = 'Classical'
        else:
            self.current_class['model_type'] = 'Class'

    def _safe_unparse(self, node: ast.AST) -> str:
        """Safely convert AST node to string - FULL LAMBDA SUPPORT."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node).strip()

            if isinstance(node, ast.Constant):
                return repr(node.value)
            if isinstance(node, (ast.Str, ast.Bytes)):
                return repr(getattr(node, 's', node.value))
            if isinstance(node, ast.List):
                elements = [self._safe_unparse(elt) for elt in node.elts]
                return '[' + ', '.join(str(e) for e in elements) + ']'
            if isinstance(node, ast.Dict):
                items = []
                for k, v in zip(node.keys, node.values):
                    key_str = self._safe_unparse(k) if k else 'None'
                    val_str = self._safe_unparse(v)
                    items.append(f"{key_str}: {val_str}")
                return '{' + ', '.join(items) + '}'
            if isinstance(node, ast.Lambda):
                args = ', '.join(arg.arg for arg in node.args.args)
                body = self._safe_unparse(node.body)
                return f'lambda {args}: {body}'
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                value = self._safe_unparse(node.value)
                return f'{value}.{node.attr}'
            if isinstance(node, ast.Call):
                func = self._safe_unparse(node.func)
                args = ', '.join(self._safe_unparse(arg) for arg in node.args)
                kwargs = ', '.join(f'{kw.arg}={self._safe_unparse(kw.value)}' for kw in node.keywords if kw.arg)
                all_args = ', '.join(filter(None, [args, kwargs]))
                return f'{func}({all_args})'
            if isinstance(node, ast.Tuple):
                elements = [self._safe_unparse(elt) for elt in node.elts]
                return '(' + ', '.join(str(e) for e in elements) + ')'
            if isinstance(node, ast.BinOp):
                left = self._safe_unparse(node.left)
                right = self._safe_unparse(node.right)
                op = self._get_op_symbol(node.op)
                return f'{left} {op} {right}'
            if isinstance(node, ast.Compare):
                left = self._safe_unparse(node.left)
                comparisons = []
                for op, comparator in zip(node.ops, node.comparators):
                    op_symbol = self._get_compare_symbol(op)
                    comp_val = self._safe_unparse(comparator)
                    comparisons.append(f'{op_symbol} {comp_val}')
                return f'{left} {" ".join(comparisons)}'
            if isinstance(node, ast.Subscript):
                value = self._safe_unparse(node.value)
                slice_val = self._safe_unparse(node.slice)
                return f'{value}[{slice_val}]'
            if isinstance(node, ast.Index):
                return self._safe_unparse(node.value)

        except Exception:
            pass

        return f'<{type(node).__name__}>'

    def _get_op_symbol(self, op):
        """Get operator symbol."""
        ops = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Mod: '%', ast.Pow: '**', ast.FloorDiv: '//'
        }
        return ops.get(type(op), '?')

    def _get_compare_symbol(self, op):
        """Get comparison operator symbol."""
        ops = {
            ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=',
            ast.Gt: '>', ast.GtE: '>=', ast.Is: 'is', ast.IsNot: 'is not',
            ast.In: 'in', ast.NotIn: 'not in'
        }
        return ops.get(type(op), '?')

    def _parse_inherit_list(self, value_str: str) -> List[str]:
        """Parse _inherit list values."""
        try:
            matches = re.findall(r"'([^']*)'", value_str)
            return [m for m in matches if m not in ['mail.thread', 'mail.activity.mixin']]
        except:
            return []

    def _parse_inherits_dict(self, value_str: str) -> Dict[str, str]:
        """Parse _inherits dictionary values."""
        try:
            import ast as ast_module
            parsed = ast_module.literal_eval(value_str)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass
        return {}

    def _extract_docstring(self, node: ast.ClassDef) -> Optional[str]:
        """Extract class docstring."""
        return ast.get_docstring(node) or ''


def analyze_file(file_path: str, module_name: str) -> List[Dict]:
    """Analyze single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=file_path)
        analyzer = OdooModelAnalyzer()
        analyzer.current_module = module_name
        analyzer.visit(tree)

        for model in analyzer.models:
            model['module'] = module_name
            model['file'] = str(Path(file_path).name)

        return analyzer.models
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
        return []


def analyze_folder(addons_path: str) -> Dict[str, Any]:
    """Scan entire addons folder."""
    report = {
        'analyzed_path': str(Path(addons_path).absolute()),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': [],
        'summary': {},
        'stats': {}
    }

    root_path = Path(addons_path)
    if not root_path.is_dir():
        print(f"Path not found: {addons_path}", file=sys.stderr)
        return report

    print(f"Scanning {addons_path}...")

    for module_path in sorted(root_path.glob('*')):
        if module_path.is_dir():
            models_path = module_path / 'models'
            if models_path.exists():
                print(f"Module: {module_path.name}")
                for py_file in sorted(models_path.rglob('*.py')):
                    models = analyze_file(str(py_file), module_path.name)
                    report['models'].extend(models)

    report['summary'] = generate_summary(report['models'])
    report['stats'] = {
        'total_models': len(report['models']),
        'total_modules': len({m['module'] for m in report['models']}),
    }

    print(f"Analysis complete: {report['stats']['total_models']} models found")
    return report


def generate_summary(models: List[Dict]) -> Dict:
    """Generate comprehensive statistics."""
    summary = {}
    totals = {
        'total_fields': 0,
        'regular_fields': 0,
        'computed_stored': 0,
        'computed_non_stored': 0,
        'related_stored': 0,
        'related_non_stored': 0,
        'related_fields': 0,
        'total_methods': 0,
        'api_model': 0,
        'api_onchange': 0,
        'api_depends': 0,
        'api_constrains': 0
    }

    for model in models:
        model_key = model['model_name'] or model['class_name']

        related_stored = len([f for f in model['related_fields'] if f.get('stored') == True])
        related_non_stored = len([f for f in model['related_fields'] if f.get('stored') != True])

        api_stats = Counter()
        for method in model['methods']:
            for decorator in method['decorators']:
                if '@api.model' in decorator:
                    api_stats['api_model'] += 1
                elif '@api.onchange' in decorator:
                    api_stats['api_onchange'] += 1
                elif '@api.depends' in decorator:
                    api_stats['api_depends'] += 1
                elif '@api.constrains' in decorator:
                    api_stats['api_constrains'] += 1

        model_stats = {
            'type': model['model_type'],
            'module': model['module'],
            'total_fields': len(model['regular_fields']) + len(model['computed_stored']) + len(
                model['computed_non_stored']) + len(model['related_fields']),
            'regular_fields': len(model['regular_fields']),
            'computed_stored': len(model['computed_stored']),
            'computed_non_stored': len(model['computed_non_stored']),
            'related_stored': related_stored,
            'related_non_stored': related_non_stored,
            'related_fields': len(model['related_fields']),
            'total_methods': len(model['methods']),
            'api_model': api_stats.get('api_model', 0),
            'api_onchange': api_stats.get('api_onchange', 0),
            'api_depends': api_stats.get('api_depends', 0),
            'api_constrains': api_stats.get('api_constrains', 0)
        }

        summary[model_key] = model_stats

        for key in totals:
            totals[key] += model_stats.get(key, 0)

    summary['TOTAL'] = totals
    return summary


def print_console_summary(report: Dict) -> None:
    """Print compact console summary table."""
    print("\n" + "=" * 120)
    print("ODOO MODULES ANALYSIS SUMMARY")
    print(f"Path: {report['analyzed_path']}")
    print(f"Generated: {report['timestamp']}")
    print(f"Models: {report['stats']['total_models']} | Modules: {report['stats']['total_modules']}")
    print("=" * 120)


def save_json_report(report: Dict, output_file: str) -> None:
    """Save structured JSON data."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"JSON data: {output_file}")


def generate_html_report(report: Dict, output_file: str = None):
    """Generate HTML report with detailed statistics."""
    modules = defaultdict(list)
    for model in report['models']:
        module = model.get('module', 'unknown')
        modules[module].append(model)

    total_summary = report['summary']['TOTAL']
    total_fields = total_summary.get('total_fields', 0)
    regular_fields = total_summary.get('regular_fields', 0)
    computed_stored = total_summary.get('computed_stored', 0)
    computed_non_stored = total_summary.get('computed_non_stored', 0)
    related_stored = total_summary.get('related_stored', 0)
    related_non_stored = total_summary.get('related_non_stored', 0)

    comp_ns_color = '#e74c3c' if computed_non_stored > 0 else '#27ae60'
    rel_s_color = '#e74c3c' if related_stored > 0 else '#27ae60'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Odoo Analysis</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',sans-serif;line-height:1.6;color:#333;max-width:1200px;margin:0 auto;padding:40px 30px;background:#fff}}
.header{{text-align:center;margin-bottom:50px;padding-bottom:30px;border-bottom:3px solid #2c3e50}}
h1{{font-size:2.2em;color:#2c3e50;margin-bottom:10px;font-weight:300}}
.meta-info{{color:#7f8c8d;font-size:0.95em;line-height:1.8}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin-top:20px;text-align:left}}
.stat-card{{background:#f8f9fa;padding:15px;border-radius:8px;border-left:4px solid #3498db}}
.stat-label{{font-size:0.85em;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.5px}}
.stat-value{{font-size:1.8em;font-weight:700;color:#2c3e50;margin:5px 0}}
.stat-detail{{font-size:0.9em;color:#95a5a6}}
.warning{{color:#e74c3c;font-weight:600}}
.success{{color:#27ae60;font-weight:600}}
h2{{color:#2c3e50;font-size:1.6em;margin:40px 0 20px 0;padding-bottom:10px;border-bottom:2px solid #ecf0f1;font-weight:400}}
h3{{color:#34495e;font-size:1.2em;margin:25px 0 15px 0;font-weight:500}}
.model-header{{background:#f8f9fa;padding:20px;border-left:5px solid #3498db;margin-bottom:25px;border-radius:0 5px 5px 0}}
.model-type{{display:inline-block;padding:6px 12px;margin-right:15px;border-radius:4px;font-size:0.85em;font-weight:600;text-transform:uppercase}}
.model-type.class{{background:#d5f4e6;color:#27ae60}}
.model-type.classical{{background:#fef9e7;color:#f39c12}}
.model-type.extension{{background:#eaf3ff;color:#3498db}}
.model-type.delegation{{background:#ffe6f0;color:#e91e63}}
.model-meta{{color:#7f8c8d;font-size:0.9em;margin-top:8px}}
ul{{list-style:none;margin:20px 0}}
li{{padding:12px 0;border-bottom:1px solid #ecf0f1}}
.field-name{{font-family:Consolas,monospace;background:#f8f9fa;padding:3px 8px;border-radius:3px;font-size:0.95em;margin-right:10px}}
.attr-bold{{font-weight:600;color:#e74c3c}}
table{{width:100%;border-collapse:collapse;margin:25px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
th{{background:#2c3e50;color:#fff;padding:15px 12px;text-align:left;font-weight:500;font-size:0.9em;text-transform:uppercase}}
td{{padding:14px 12px;border-bottom:1px solid #ecf0f1}}
tr:hover{{background:#f8f9fa}}
.super-yes{{color:#27ae60;font-weight:600}}
.super-no{{color:#95a5a6}}
.summary-box{{background:#f8f9fa;border-left:5px solid #3498db;padding:25px;margin:30px 0;border-radius:0 5px 5px 0}}
.summary-title{{font-size:1.1em;color:#2c3e50;margin-bottom:15px;font-weight:600}}
.citation{{color:#95a5a6;font-size:0.85em}}
hr{{border:none;height:2px;background:linear-gradient(to right,transparent,#bdc3c7,transparent);margin:50px 0}}
.total-row{{background:#ecf0f1!important;font-weight:700;font-size:1.05em}}
.total-row td{{border-top:3px solid #3498db}}
</style></head><body>
<div class="header">
<h1>FSF Modules Analysis Report</h1>
<div class="meta-info">
<div><strong>Generated:</strong> {report.get('timestamp')}</div>
<div><strong>Analyzed:</strong> {report['analyzed_path']}</div>
</div>

<div class="stats-grid">
<div class="stat-card">
<div class="stat-label">Modules</div>
<div class="stat-value">{report['stats']['total_modules']}</div>
</div>

<div class="stat-card">
<div class="stat-label">Models</div>
<div class="stat-value">{len(report['models'])}</div>
</div>

<div class="stat-card">
<div class="stat-label">Total Fields</div>
<div class="stat-value">{total_fields}</div>
<div class="stat-detail">Regular: {regular_fields}</div>
</div>

<div class="stat-card">
<div class="stat-label">Computed Fields</div>
<div class="stat-value">{computed_stored + computed_non_stored}</div>
<div class="stat-detail">
Stored: <span class="success">{computed_stored}</span> | 
Non-Stored: <span style="color:{comp_ns_color}">{computed_non_stored}</span>
</div>
</div>

<div class="stat-card">
<div class="stat-label">Related Fields</div>
<div class="stat-value">{related_stored + related_non_stored}</div>
<div class="stat-detail">
Stored: <span style="color:{rel_s_color}">{related_stored}</span> | 
Non-Stored: <span class="success">{related_non_stored}</span>
</div>
</div>

<div class="stat-card">
<div class="stat-label">Methods</div>
<div class="stat-value">{total_summary.get('total_methods', 0)}</div>
<div class="stat-detail">
@api.model: {total_summary.get('api_model', 0)} | 
@api.depends: {total_summary.get('api_depends', 0)} |
@api.onchange: {total_summary.get('api_onchange', 0)} |
@api.constrains: {total_summary.get('api_constrains', 0)}
</div>
</div>
</div>

</div>"""

    for module_name in sorted(modules.keys()):
        html += f'<h2>Module: {module_name}</h2>'

        for model in modules[module_name]:
            model_name = model['model_name'] or (model['inherit'][0] if model['inherit'] else model['class_name'])
            file_path = model.get('file', 'unknown')

            type_class = model['model_type'].lower().replace(' ', '-')
            html += f'<div class="model-header"><h3>{module_name}/{model_name}</h3>'
            html += f'<span class="model-type {type_class}">{model["model_type"]}</span>'
            html += '<div class="model-meta">'
            if model['model_name']:
                html += f'<code>_name: {model["model_name"]}</code><br>'
            if model['inherit']:
                html += f'<code>_inherit: {", ".join(model["inherit"])}</code><br>'
            if model.get('inherits'):
                inherits_str = ', '.join([f'{k} via {v}' for k, v in model['inherits'].items()])
                html += f'<code>_inherits: {inherits_str}</code><br>'
            html += f'<span class="citation">[{file_path}]</span></div></div>'

            if model['regular_fields']:
                html += f'<h3>Regular Fields ({len(model["regular_fields"])})</h3><ul>'
                for f in model['regular_fields']:
                    attrs = ' '.join([f'<span class="attr-bold">{k}</span>={v}' for k, v in f['attrs'].items()])
                    html += f'<li><span class="field-name">{f["name"]}</span>: {f["type"]} ({attrs})</li>'
                html += '</ul>'

            if model['computed_stored'] or model['computed_non_stored']:
                html += '<h3>Computed Fields</h3>'
                if model['computed_stored']:
                    html += f'<p><strong>Stored ({len(model["computed_stored"])})</strong></p><ul>'
                    for f in model['computed_stored']:
                        attrs = ' '.join([f'<span class="attr-bold">{k}</span>={v}' for k, v in f['attrs'].items()])
                        html += f'<li><span class="field-name">{f["name"]}</span>: {f["type"]} ({attrs})</li>'
                    html += '</ul>'
                if model['computed_non_stored']:
                    html += f'<p><strong>Non-Stored ({len(model["computed_non_stored"])})</strong></p><ul>'
                    for f in model['computed_non_stored']:
                        attrs = ' '.join([f'<span class="attr-bold">{k}</span>={v}' for k, v in f['attrs'].items()])
                        html += f'<li><span class="field-name">{f["name"]}</span>: {f["type"]} ({attrs})</li>'
                    html += '</ul>'

            if model['related_fields']:
                related_stored = [f for f in model['related_fields'] if f.get('stored') == True]
                related_non_stored = [f for f in model['related_fields'] if f.get('stored') != True]

                html += f'<h3>Related Fields ({len(model["related_fields"])})</h3>'

                if related_stored:
                    html += f'<p><strong>Stored ({len(related_stored)})</strong></p><ul>'
                    for f in related_stored:
                        attrs = ' '.join([f'<span class="attr-bold">{k}</span>={v}' for k, v in f['attrs'].items()])
                        html += f'<li><span class="field-name">{f["name"]}</span>: {f["type"]} ({attrs})</li>'
                    html += '</ul>'

                if related_non_stored:
                    html += f'<p><strong>Non-Stored ({len(related_non_stored)})</strong></p><ul>'
                    for f in related_non_stored:
                        attrs = ' '.join([f'<span class="attr-bold">{k}</span>={v}' for k, v in f['attrs'].items()])
                        html += f'<li><span class="field-name">{f["name"]}</span>: {f["type"]} ({attrs})</li>'
                    html += '</ul>'

            html += '<h3>Methods</h3><table><thead><tr><th>Method name</th><th>Decorators</th><th>Has super</th></tr></thead><tbody>'
            for m in model['methods']:
                decors = '; '.join(m['decorators']) if m['decorators'] else ''
                super_class = 'super-yes' if m['has_super'] else 'super-no'
                super_text = '✅ yes' if m['has_super'] else '❌ no'
                html += f'<tr><td><span class="field-name">{m["name"]}</span></td><td>{decors}</td><td class="{super_class}">{super_text}</td></tr>'
            html += '</tbody></table>'

            summ = report['summary'].get(model_name, {})
            comp_total = summ.get('computed_stored', 0) + summ.get('computed_non_stored', 0)
            comp_stored = summ.get('computed_stored', 0)
            comp_non_stored = summ.get('computed_non_stored', 0)
            related_total = summ.get('related_fields', 0)
            rel_stored = summ.get('related_stored', 0)
            rel_non_stored = summ.get('related_non_stored', 0)

            comp_non_stored_html = f'<strong>{comp_non_stored}</strong>' if comp_non_stored == 0 else f'<strong style="color:#e74c3c">{comp_non_stored}</strong>'
            related_stored_html = f'<strong>{rel_stored}</strong>' if rel_stored == 0 else f'<strong style="color:#e74c3c">{rel_stored}</strong>'

            html += f'''<div class="summary-box"><div class="summary-title">📊 Model Summary</div>
<strong>{model["model_type"]}</strong> | Total Fields: <strong>{summ.get("total_fields", 0)}</strong> | 
Computed: <strong>{comp_total}</strong> (Stored: <strong>{comp_stored}</strong>, Non-Stored: {comp_non_stored_html}) | 
Related: <strong>{related_total}</strong> (Stored: {related_stored_html}, Non-Stored: <strong>{rel_non_stored}</strong>) | 
Methods: <strong>{summ.get("total_methods", 0)}</strong> 
(<strong>@api.model:</strong> {summ.get("api_model", 0)}, 
<strong>@api.onchange:</strong> {summ.get("api_onchange", 0)}, 
<strong>@api.depends:</strong> {summ.get("api_depends", 0)}, 
<strong>@api.constrains:</strong> {summ.get("api_constrains", 0)})
</div>'''
            html += '<hr>'

    html += '<h1>Complete Summary</h1><table><thead><tr>'
    html += '<th>Model</th><th>Type</th><th>Total Fields</th><th>Computed (S/NS)</th><th>Related (S/NS)</th><th>Methods</th><th>model</th><th>onchange</th><th>depends</th><th>constrains</th>'
    html += '</tr></thead><tbody>'

    total = report['summary']['TOTAL']
    for model_name, stats in report['summary'].items():
        if model_name == 'TOTAL':
            continue
        comp_str = f"{stats['computed_stored'] + stats['computed_non_stored']} ({stats['computed_stored']}/{stats['computed_non_stored']})"
        rel_str = f"{stats['related_stored'] + stats['related_non_stored']} ({stats['related_stored']}/{stats['related_non_stored']})"
        html += f"<tr><td>{model_name}</td><td>{stats['type']}</td><td>{stats['total_fields']}</td><td>{comp_str}</td><td>{rel_str}</td><td>{stats['total_methods']}</td><td>{stats['api_model']}</td><td>{stats['api_onchange']}</td><td>{stats['api_depends']}</td><td>{stats['api_constrains']}</td></tr>"

    comp_total = total['computed_stored'] + total['computed_non_stored']
    rel_total = total['related_stored'] + total['related_non_stored']
    html += f"<tr class='total-row'><td><strong>TOTAL</strong></td><td></td><td><strong>{total['total_fields']}</strong></td><td><strong>{comp_total} ({total['computed_stored']}/{total['computed_non_stored']})</strong></td><td><strong>{rel_total} ({total['related_stored']}/{total['related_non_stored']})</strong></td><td><strong>{total['total_methods']}</strong></td><td><strong>{total['api_model']}</strong></td><td><strong>{total['api_onchange']}</strong></td><td><strong>{total['api_depends']}</strong></td><td><strong>{total['api_constrains']}</strong></td></tr>"
    html += '</tbody></table></body></html>'

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML report saved: {output_file}")

    return html


def main():
    parser = argparse.ArgumentParser(description="🚀 Professional Odoo Model Analyzer")
    parser.add_argument('addons_path', help='Path to custom addons folder')
    parser.add_argument('-o', '--html', help='HTML output file')
    parser.add_argument('-j', '--json', help='JSON data file')
    parser.add_argument('--console-only', action='store_true', help='Console output only')

    args = parser.parse_args()

    report = analyze_folder(args.addons_path)
    print_console_summary(report)

    if args.console_only:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_file = args.html or f"odoo_report_{timestamp}.html"
    generate_html_report(report, html_file)

    if args.json:
        save_json_report(report, args.json)


if __name__ == '__main__':
    main()
