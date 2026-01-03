
import re
from pathlib import Path

# --- Patterns for Algorithmic Cleaning ---
CAMEL_SPLIT = re.compile(r'([a-z])([A-Z])')
LETTER_DIGIT = re.compile(r'([a-zA-Z])(\d)')
DIGIT_LETTER = re.compile(r'(\d)([a-zA-Z])')
SUFFIXES = r'(Shan|Hu|Dao|Jiang|Xing|Ce|Diao|Biao|vessels|ships|forces|auxiliarios)'
SUFFIX_RE = re.compile(r'([a-zA-Z])' + SUFFIXES + r'\b', re.IGNORECASE)
CLEAN_PUNCT = re.compile(r'(^[-. ]+|[-. ]+$)')

def algorithmic_clean_text(text):
    """Programmatically fixes spacing/spelling artifacts."""
    if not text.strip(): return ""
    # E.g., 'EmeiShan' -> 'Emei Shan'
    text = CAMEL_SPLIT.sub(r'\1 \2', text)
    # E.g., 'Biao265' -> 'Biao 265'
    text = LETTER_DIGIT.sub(r'\1 \2', text)
    text = DIGIT_LETTER.sub(r'\1 \2', text)
    # E.g., 'QiandaoHu' -> 'Qiandao Hu'
    text = SUFFIX_RE.sub(r'\1 \2', text)
    
    words = text.split()
    cleaned = [CLEAN_PUNCT.sub('', w) for w in words]
    return " ".join(cleaned).strip()

def process_organized_table(table_lines):
    """Cleans text inside cells but keeps the column structure exactly the same."""
    processed_rows = []
    for line in table_lines:
        if re.match(r'^\s*\|[\s:-|]+\|\s*$', line):
            processed_rows.append(line) # Keep separator |---| as is
            continue
        
        cells = line.strip().strip('|').split('|')
        cleaned_cells = [f" {algorithmic_clean_text(c)} " for c in cells]
        processed_rows.append("|" + "|".join(cleaned_cells) + "|\n")
    return processed_rows

def restructure_messy_table(table_lines):
    """Converts a wide multi-column table into a 2-column 'Hull No. | Name' table."""
    new_rows = [
        "| Hull No.        | Name                      |",
        "|-----------------|---------------------------|"
    ]
    for line in table_lines:
        if re.match(r'^\s*\|[\s:-|]+\|\s*$', line): continue
        
        # Extract all cells
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        
        # Pair them up (Column 1+2, 3+4, 5+6...)
        for i in range(0, len(cells), 2):
            hull = cells[i] if i < len(cells) else ""
            name = cells[i+1] if i+1 < len(cells) else ""
            if hull or name:
                new_rows.append(f"| {algorithmic_clean_text(hull):<15} | {algorithmic_clean_text(name):<25} |")
    return [r + "\n" for r in new_rows]

def clean_markdown(input_md, output_md):
    lines = Path(input_md).read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    i = 0

    while i < len(lines):
        # Check if line is a table
        if lines[i].strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            
            # Count columns in the first non-separator row
            sample_row = table_lines[0]
            column_count = sample_row.count("|") - 1
            
            # DECISION LOGIC:
            # If the table is very wide (> 6 columns), it's the 'messy' one to be restructured.
            # If it's a standard table (like the Builders table), preserve structure.
            if column_count > 6:
                out.extend(restructure_messy_table(table_lines))
            else:
                out.extend(process_organized_table(table_lines))
            continue
        
        # Keep non-table lines exactly as they are
        out.append(lines[i])
        i += 1

    Path(output_md).write_text("".join(out), encoding="utf-8")

if __name__ == "__main__":
    clean_markdown("final_perfect_extraction.md", "cleaned_final_output.md")

