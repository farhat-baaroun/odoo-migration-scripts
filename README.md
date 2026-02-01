# Odoo Migration Scripts

Professional Odoo Model Analyzer - Analyze Odoo modules and generate comprehensive HTML reports with detailed statistics about models, fields, and methods.

## Features

- 🔍 **Comprehensive Analysis**: Scans Odoo models, fields (regular, computed, related), and methods
- 📊 **Detailed Reports**: Generates interactive HTML reports with collapsible sections
- 🎯 **Easy Navigation**: All elements have IDs and data attributes for easy JavaScript querying
- 📈 **Statistics**: Summary tables with field counts, method decorators, and inheritance types
- 🔄 **Model Classification**: Automatically classifies models as Class, Classical, Extension, or Delegation

## Setup

1. Clone this repository
2. Ensure you have Python 3.6+ installed
3. No additional dependencies required (uses only standard library)

## Usage

### Basic Workflow

1. **Place your Odoo modules** in the `addons_folder/` directory:
   ```
   addons_folder/
   ├── module1/
   │   └── models/
   │       ├── model1.py
   │       └── model2.py
   ├── module2/
   │   └── models/
   │       └── model3.py
   └── ...
   ```

2. **Run the script**:
   ```bash
   python module_report.py addons_folder
   ```

3. **View the report**: The HTML report will be automatically saved in `addons_reports/` directory with a timestamped filename (e.g., `odoo_report_20241201_143022.html`)

### Command-Line Options

```bash
# Basic usage - scans addons_folder and saves to addons_reports/
python module_report.py addons_folder

# Specify custom HTML output filename
python module_report.py addons_folder -o addons_reports/my_report.html

# Generate both HTML and JSON files
python module_report.py addons_folder -o addons_reports/report.html -j addons_reports/report.json

# Console output only (no HTML file)
python module_report.py addons_folder --console-only
```

### Arguments

- `addons_path` (required): Path to your addons folder (typically `addons_folder`)
- `-o, --html`: Custom HTML output file path
- `-j, --json`: Also save JSON data file
- `--console-only`: Only print summary to console (no HTML file)

## Output

### HTML Report Features

- **Collapsible Models**: All models are collapsed by default for easier navigation
- **Interactive Elements**: All sections, fields, and methods have unique IDs and data attributes
- **Statistics Dashboard**: Overview cards showing totals for modules, models, fields, and methods
- **Detailed Model View**: Each model shows:
  - Model type (Class, Classical, Extension, Delegation)
  - Regular fields with attributes
  - Computed fields (stored/non-stored)
  - Related fields (stored/non-stored)
  - Methods with decorators and super() call detection
  - Model summary statistics

### JavaScript Query Examples

The HTML report includes IDs and data attributes for easy querying:

```javascript
// Get a specific model
document.querySelector('#model-res_partner')

// Get all fields named "name" across all models
document.querySelectorAll('[data-field-name="name"]')

// Get all computed non-stored fields
document.querySelectorAll('[data-computed="true"][data-stored="false"]')

// Get all methods with @api.model decorator
document.querySelectorAll('[data-decorator*="@api.model"]')

// Get all models of type "Class"
document.querySelectorAll('[data-model-type="Class"]')
```

## Directory Structure

```
odoo-migration-scripts/
├── module_report.py      # Main script
├── README.md             # This file
├── addons_folder/        # Place your Odoo modules here (gitignored)
│   ├── module1/
│   │   └── models/
│   └── module2/
│       └── models/
└── addons_reports/       # Generated reports (gitignored)
    ├── odoo_report_20241201_143022.html
    └── ...
```

## Model Classification

The script automatically classifies Odoo models:

- **Class**: New model with `_name` (no significant inheritance)
- **Classical**: Model with `_name` and `_inherit` (extends another model)
- **Extension**: Model without `_name` but with `_inherit` (pure extension)
- **Delegation**: Model with `_inherits` (delegation inheritance)

## Field Analysis

The script analyzes:

- **Regular Fields**: Standard Odoo fields
- **Computed Fields**: Fields with `compute` attribute
  - Stored vs Non-Stored (non-stored are highlighted as warnings)
- **Related Fields**: Fields with `related` attribute
  - Stored vs Non-Stored (stored are highlighted as warnings)

## Method Analysis

For each method, the script detects:

- **Decorators**: `@api.model`, `@api.depends`, `@api.onchange`, `@api.constrains`
- **Super Calls**: Whether the method calls `super()` (important for inheritance)

## Notes

- The `addons_folder/` and `addons_reports/` directories are gitignored
- Reports are timestamped to avoid overwriting previous analyses
- The script recursively scans all `.py` files in `models/` subdirectories
- Only files containing Odoo model classes (inheriting from `models.Model`) are analyzed

