Update performance metrics in documentation from evaluation results.

Read the latest evaluation results from `make eval` and update all documentation files with current model performance metrics for both pieces and arrows models.

## Prerequisites

Run evaluation first:
```bash
make eval
```

This generates JSON files with metrics in `evals/` directory.

## Workflow

### 1. Read Evaluation Results

Read all JSON files in parallel:

**Pieces Model:**
- `evals/pieces/test/test_summary.json`
- `evals/pieces/openboard/test_summary.json`
- `evals/pieces/chessvision/test_summary.json`

**Arrows Model:**
- `evals/arrows/test/test_summary.json`

From each, extract:
- `overall_accuracy`
- `f1_score_macro`
- `per_class_accuracy` (dict)
- `num_test_samples`

### 2. Update Files Using Pattern Matching

**IMPORTANT:** Use content-based pattern matching, NOT line numbers (files may change over time).

#### README.md

**Pattern 1 - Pieces Model Table:**
Search for table containing:
```
| Test Data                                                                                       |  99.94%  |      99.94%      |
| [S1M0N38/chess-cv-openboard]
```

Update:
- Test Data row: accuracy (2 decimals, %), F1-Score (2 decimals, %)
- OpenBoard row: keep "-" for accuracy, update F1-Score (2 decimals, %)
- ChessVision row: keep "-" for accuracy, update F1-Score (2 decimals, %)

**Pattern 2 - Arrows Model Table:**
Search for table containing:
```
| Test Data (synthetic) |  99.97%  |      99.97%      |
```

Update Test Data row: accuracy (2 decimals, %), F1-Score (2 decimals, %)

#### docs/README_hf.md

**Pattern 1 - YAML Frontmatter (Test Dataset):**
Search for:
```yaml
        dataset:
          name: Chess CV Test Dataset
          type: chess-cv-test
        metrics:
          - type: accuracy
            value: 0.9994
```

Update accuracy `value` and f1 `value` (4 decimals, decimal format not %).

**Pattern 2 - YAML Frontmatter (OpenBoard Dataset):**
Search for:
```yaml
        dataset:
          name: Chess CV OpenBoard Dataset
          type: chess-cv-openboard
```

Update accuracy `value` and f1 `value` in the metrics section below.

**Pattern 3 - YAML Frontmatter (ChessVision Dataset):**
Search for:
```yaml
        dataset:
          name: Chess CV ChessVision Dataset
          type: chess-cv-chessvision
```

Update accuracy `value` and f1 `value` in the metrics section below.

**Pattern 4 - YAML Frontmatter (Arrows Dataset):**
Search for:
```yaml
  - name: chess-cv-arrows
    results:
      - task:
```

Update accuracy `value` and f1 `value` in the metrics section below.

**Pattern 5 & 6 - Performance Tables:**
Same as README.md tables - use same search patterns.

#### docs/architecture.md

**Pattern 1 - Pieces Test Performance:**
Search for:
```markdown
- **Test Accuracy**: ~99.94%
- **F1 Score (Macro)**: ~99.94%
```

Update both values (use "~" prefix, 2 decimals, %).

**Pattern 2 - Pieces Per-Class Test Table:**
Search for table with header:
```markdown
| Class | Accuracy | Class | Accuracy |
| ----- | -------- | ----- | -------- |
| bB    |
```

Update all 13 classes from `per_class_accuracy` dict (2 decimals, %).

**Pattern 3 - OpenBoard Overall:**
Search for:
```markdown
- **Overall Accuracy**: 99.30%
- **F1 Score (Macro)**: 98.26%
```

Update both values (2 decimals, %).

**Pattern 4 - OpenBoard Per-Class Table:**
Search for table after OpenBoard section with same structure as test table.
Update all 13 classes (2 decimals, %).

**Pattern 5 - ChessVision Overall:**
Search for:
```markdown
- **Overall Accuracy**: 86.38%
- **F1 Score (Macro)**: 83.47%
```

Update both values (2 decimals, %).

**Pattern 6 - ChessVision Per-Class Table:**
Search for table after ChessVision section with same structure.
Update all 13 classes (2 decimals, %).

**Pattern 7 - Comparison Note:**
Search for paragraph starting with:
```markdown
The lower performance on OpenBoard (
```

Update all 6 metrics in this sentence:
- OpenBoard: accuracy%, F1%
- ChessVision: accuracy%, F1%
- Test set: accuracy%, F1%

All formatted as 2 decimals with %.

**Pattern 8 - Arrows Test Performance:**
Search for:
```markdown
- **Test Accuracy**: ~99.97%
- **F1 Score (Macro)**: ~99.97%
```

In Arrows Model section, update both (use "~" prefix, 2 decimals, %).

**Pattern 9 - Arrows Per-Class Summary:**
Search for:
```markdown
- **Highest Accuracy**: 100.00%
- **Lowest Accuracy**: 99.79%
- **Mean Accuracy**: 99.97%
- **Classes > 99.9%**: 44 out of 49
```

Calculate from `per_class_accuracy` dict:
- Highest: max value
- Lowest: min value
- Mean: average of all values
- Classes > 99.9%: count where value > 0.999

**Pattern 10 - Arrows Performance by Component:**
Search for section:
```markdown
**Performance by Component Type:**

| Component Type  | Classes | Avg Accuracy | Range         |
| --------------- | ------- | ------------ | ------------- |
| Arrow Heads     |
```

Calculate for each component type by grouping classes:
- Arrow Heads: prefix "head-" (count, avg, min-max range)
- Arrow Tails: prefix "tail-" (count, avg, min-max range)
- Middle Segments: prefix "middle-" (count, avg, min-max range)
- Corners: prefix "corner-" (count, avg, min-max range)
- Empty Square: class "xx" (just show accuracy)

#### AGENTS.md

**Pattern 1 - Project Overview:**
Search for:
```markdown
Chess-CV is a CNN-based chess piece classifier that uses MLX (Apple's ML framework) to train a lightweight 156k parameter model. The model classifies 32×32px square images into 13 classes (6 white pieces, 6 black pieces, 1 empty square) with ~99.85% accuracy.
```

Update accuracy value at the end (use "~" prefix, 2 decimals, %) from pieces test dataset.

### 3. Formatting Rules

1. **Markdown tables/text**: 2 decimal places as percentage (e.g., 99.94%)
2. **YAML frontmatter values**: 4 decimal places as decimal (e.g., 0.9994)
3. **Architecture.md approximations**: Use "~" prefix (e.g., ~99.94%)
4. **Special cases**: OpenBoard and ChessVision accuracy in main tables show "-" (due to class imbalance)

### 4. Validation

After updates:

1. **Verify JSON files are valid:**
   - Check all required JSON files exist
   - Validate JSON structure
   - Confirm metric values in range [0, 1]

2. **Verify updates were applied:**
   - Search for old metric values
   - Confirm none remain

3. **Check formatting:**
   - YAML frontmatter: 4 decimals, decimal format
   - Markdown: 2 decimals, percentage format
   - Tables: alignment preserved
   - Special characters: "-" preserved where needed

4. **Validate structure:**
   - YAML indentation correct (2 spaces)
   - Table separators intact
   - All per-class accuracies updated

### 5. Report Changes

Generate summary showing:

**Pieces Model:**
- Test: [old_acc → new_acc]%, [old_f1 → new_f1]%
- OpenBoard: [old_f1 → new_f1]%
- ChessVision: [old_f1 → new_f1]%

**Arrows Model:**
- Test: [old_acc → new_acc]%, [old_f1 → new_f1]%

**Files Modified:**
- README.md
- docs/README_hf.md
- docs/architecture.md
- AGENTS.md

## Error Handling

**Missing evaluation files:**
→ Prompt user to run `make eval` first

**Invalid JSON:**
→ Report error with file path

**Metrics out of range:**
→ Warn if value < 0 or > 1

**Wrong class count:**
→ Error if pieces ≠ 13 classes or arrows ≠ 49 classes

## Data Structure Reference

**Pieces Model JSON** (`evals/pieces/{dataset}/test_summary.json`):
```json
{
  "overall_accuracy": 0.9994,
  "f1_score_macro": 0.9994,
  "per_class_accuracy": {
    "bB": 0.9990, "bK": 1.0, "bN": 1.0, "bP": 0.9990,
    "bQ": 0.9990, "bR": 1.0, "wB": 0.9990, "wK": 0.9981,
    "wN": 1.0, "wP": 0.9990, "wQ": 1.0, "wR": 1.0, "xx": 0.9990
  },
  "num_test_samples": 13574
}
```

**Arrows Model JSON** (`evals/arrows/test/test_summary.json`):
```json
{
  "overall_accuracy": 0.9997,
  "f1_score_macro": 0.9997,
  "per_class_accuracy": {
    "corner-E-S": 1.0, "corner-N-E": 0.9999,
    ... (49 classes total)
  },
  "num_test_samples": 672594
}
```

## Notes

- Run after `make eval` completes
- Updates both pieces and arrows models
- Pieces: 3 datasets (test, openboard, chessvision)
- Arrows: 1 dataset (test only)
- Use pattern matching, not line numbers
- Preserve formatting and structure
- ChessVision uses concatenated splits
