#!/usr/bin/env python3
"""Parse evaluation.txt (YAML inside markdown code fences) into evaluation.json.
Handles YAML parsing errors by fixing known issues."""

import json
import re
import yaml

def fix_yaml_block(block):
    """Fix YAML values that contain unquoted colons and other issues."""
    lines = block.split('\n')
    fixed = []
    for line in lines:
        # Match lines where value contains a colon (common in error messages, questions, answers)
        m = re.match(r'^(\s+(?:answer|question|approach|rejection_reason|description|information|decision|previous_requirement|updated_requirement|reason):\s)(.*:.*)$', line)
        if m:
            prefix = m.group(1)
            value = m.group(2).strip()
            if not (value.startswith('"') and value.endswith('"')) and not (value.startswith("'") and value.endswith("'")):
                # Escape internal quotes and wrap
                value = value.replace('"', '\\"')
                fixed.append(f'{prefix}"{value}"')
                continue
        fixed.append(line)
    return '\n'.join(fixed)

def parse_evaluation(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find Part boundaries
    part_pattern = re.compile(r'(?:PART|Part)\s*([A-Da-d])\s*:', re.IGNORECASE)
    yaml_pattern = re.compile(r'```yaml\s*\n(.*?)```', re.DOTALL)

    # Get all part positions
    part_positions = [(m.start(), m.group(1).upper()) for m in part_pattern.finditer(text)]
    
    # Get all YAML blocks with positions
    yaml_matches = [(m.start(), m.group(1)) for m in yaml_pattern.finditer(text)]

    result = {"part_a": {}, "part_b": {}, "part_c": {}, "part_d": {}}

    for block_pos, block_text in yaml_matches:
        # Determine which part this block belongs to
        current_part = "A"
        for pos, part_letter in part_positions:
            if pos < block_pos:
                current_part = part_letter

        part_key = f"part_{current_part.lower()}"
        fixed_block = fix_yaml_block(block_text)
        
        try:
            parsed = yaml.safe_load(fixed_block)
            if isinstance(parsed, dict):
                result[part_key].update(parsed)
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse YAML block in {part_key}: {e}")
            # Try line-by-line recovery for complex blocks
            print(f"  Attempting manual recovery...")
            try:
                # Split into sub-sections by #### separator
                sections = re.split(r'#{10,}', fixed_block)
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    try:
                        parsed = yaml.safe_load(section)
                        if isinstance(parsed, dict):
                            result[part_key].update(parsed)
                    except:
                        pass
            except:
                pass
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary
    print(f"✅ Parsed evaluation.txt into {output_file}")
    for part_name in ["part_a", "part_b", "part_c", "part_d"]:
        part_data = result[part_name]
        if part_data:
            print(f"\n   {part_name.upper()}:")
            for key, val in part_data.items():
                if isinstance(val, list):
                    print(f"     {key}: {len(val)} items")
                elif isinstance(val, dict):
                    print(f"     {key}: {len(val)} keys")
                else:
                    desc = str(val)[:60]
                    print(f"     {key}: {desc}")

if __name__ == "__main__":
    parse_evaluation('research/evaluation.txt', 'research/evaluation.json')
