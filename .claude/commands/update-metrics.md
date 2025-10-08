Update performance metrics in documentation from evaluation results.

Read the latest evaluation results from `make eval` and update all documentation files with the current model performance metrics.

### Workflow

1. **Read Evaluation Results**
   - Read `evals/pieces/test/test_summary.json` for test dataset results
   - Read `evals/pieces/openboard/test_summary.json` for OpenBoard dataset results
   - Read `evals/pieces/chessvision/test_summary.json` for ChessVision dataset results
   - Extract overall_accuracy, f1_score_macro, and per_class_accuracy for all three datasets

2. **Format Metrics**
   - Convert decimal values to percentages (multiply by 100)
   - Format with 2 decimal places for display (e.g., 0.9985 → 99.85%)
   - Format with 4 decimal places for YAML frontmatter (e.g., 0.9985 → 0.9985)
   - Handle per-class accuracies for all 13 classes

3. **Update README.md**
   - Locate the performance table (around lines 37-41)
   - Update Test Data row: accuracy and F1-Score columns
   - Update OpenBoard row: keep accuracy as "-" (due to class imbalance), update F1-Score
   - Update ChessVision row: keep accuracy as "-" (due to class imbalance), update F1-Score
   - Maintain table formatting and alignment

4. **Update docs/README_hf.md**
   - **YAML Frontmatter (lines 12-59)**:
     - Update first metric block (Test Dataset):
       - Line ~23: `value` for accuracy (4 decimal places)
       - Line ~27: `value` for f1 (4 decimal places)
     - Update second metric block (OpenBoard Dataset):
       - Line ~38: `value` for accuracy (4 decimal places)
       - Line ~42: `value` for f1 (4 decimal places)
     - Update third metric block (ChessVision Dataset):
       - Line ~52: `value` for accuracy (4 decimal places)
       - Line ~56: `value` for f1 (4 decimal places)
   - **Performance Table (lines 71-75)**:
     - Update Test Data row: accuracy and F1-Score columns
     - Update OpenBoard row: keep accuracy as "-" (due to class imbalance), update F1-Score
     - Update ChessVision row: keep accuracy as "-" (due to class imbalance), update F1-Score

5. **Update docs/architecture.md**
   - **Overall Test Performance (lines 89-90)**:
     - Update Test Accuracy value (e.g., "~99.85%")
     - Update F1 Score (Macro) value (e.g., "~99.89%")

   - **Per-Class Test Performance Table (lines 127-138)**:
     - Update accuracy for each class: bB, bK, bN, bP, bQ, bR, wB, wK, wN, wP, wQ, wR, xx
     - Format as percentages with 2 decimal places
     - Maintain table structure with two columns showing black pieces vs white pieces

   - **Overall OpenBoard Performance (lines 147-148)**:
     - Update Overall Accuracy value (e.g., "98.89%")
     - Update F1 Score (Macro) value (e.g., "97.25%")

   - **Per-Class OpenBoard Performance Table (lines 152-160)**:
     - Update accuracy for all 13 classes
     - Format as percentages with 2 decimal places
     - Maintain table structure

   - **Overall ChessVision Performance (lines 166-167)**:
     - Update Overall Accuracy value (e.g., "86.85%")
     - Update F1 Score (Macro) value (e.g., "83.83%")

   - **Per-Class ChessVision Performance Table (lines 171-179)**:
     - Update accuracy for all 13 classes
     - Format as percentages with 2 decimal places
     - Maintain table structure

   - **Comparison Note (lines 185-187)**:
     - Update the performance comparison text with current values for all three datasets
     - Format: "The lower performance on OpenBoard (X.XX% accuracy, Y.YY% F1) and ChessVision (A.AA% accuracy, B.BB% F1) compared to the test set (C.CC% accuracy, D.DD% F1)..."

6. **Validation**
   - Verify all JSON files exist and are valid
   - Confirm all metric values are reasonable (0 < value <= 1)
   - Ensure table formatting is preserved
   - Check that YAML frontmatter remains valid after updates
   - Verify percentage formatting is consistent across all files

7. **Report Changes**
   - List all files modified
   - Show old vs new values for key metrics
   - Confirm successful update

### Data Source Structure

The evaluation JSON files have this structure:
```json
{
  "overall_accuracy": 0.9985,
  "f1_score_macro": 0.9989,
  "per_class_accuracy": {
    "bB": 0.9982,
    "bK": 0.9982,
    "bN": 0.9973,
    "bP": 0.9982,
    "bQ": 1.0,
    "bR": 0.9964,
    "wB": 0.9991,
    "wK": 0.9954,
    "wN": 0.9991,
    "wP": 1.0,
    "wQ": 0.9982,
    "wR": 1.0,
    "xx": 1.0
  },
  "checkpoint_path": "checkpoints/pieces/best_model.safetensors",
  "test_dir": "...",
  "num_test_samples": 14014
}
```

### Formatting Rules

1. **Percentages in markdown tables/text**: 2 decimal places (99.85%)
2. **YAML frontmatter values**: 4 decimal places as decimal (0.9985)
3. **Special case**: OpenBoard and ChessVision accuracy in main tables show "-" (not percentage) due to unbalanced class distributions
4. **Tilde prefix**: Use "~" before percentages in architecture.md for approximate values (e.g., "~99.85%")
5. **Class names**: Maintain exact order as they appear in JSON (alphabetically sorted)

### Error Handling

- If evaluation files don't exist, prompt user to run `make eval` first
- If JSON is malformed, report error and exit
- If metric values are out of range (< 0 or > 1), report warning
- If class names don't match expected 13 classes, report error

### Notes

- This command should be run after `make eval` completes successfully
- The command updates metrics for the "pieces" model only (current model)
- ChessVision evaluation concatenates all splits into a single score
- All numeric formatting must preserve decimal precision for accuracy
- Table alignment should be preserved using markdown table syntax
- YAML frontmatter must remain valid (proper indentation and structure)
