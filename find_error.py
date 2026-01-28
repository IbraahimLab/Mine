
import json

with open("d:/Mine/Ch1/Notebooks/01-working_text_data.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"Cell {i} (exec {cell.get('execution_count')}):")
        source = "".join(cell['source'])
        lines = source.split('\n')
        if len(lines) >= 30:
             print(f"  Lines: {len(lines)}")
             # Print lines around 33
             for ln_idx, line in enumerate(lines):
                 if 30 <= ln_idx + 1 <= 40:
                     print(f"    {ln_idx + 1}: {line}")
