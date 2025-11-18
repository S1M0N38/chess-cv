<figure markdown="span">
  ![Chess CV](assets/logo.svg){ width="256" }
  <figcaption>Lightweight CNNs for chess board analysis</figcaption>
</figure>

---

Chess CV is a machine learning project that trains lightweight CNNs (156k parameters each) from scratch to classify different aspects of chess board squares. The project includes three specialized models trained on synthetically generated data from 55 board styles combined with piece sets and overlays from chess.com and lichess:

- **Pieces Model** (13 classes): Classifies chess pieces and empty squares for board state recognition and FEN generation
- **Arrows Model** (49 classes): Classifies arrow annotation components for detecting and reconstructing chess analysis overlays
- **Snap Model** (2 classes): Classifies piece centering quality for automated board analysis and positioning validation

Each model uses the same efficient CNN architecture but is optimized for its specific classification task, achieving robust recognition across various visual styles.

<div class="grid cards" markdown>

- :material-cog:{ .lg .middle } __Setup__

    ---

    Installation guide covering dependencies and environment setup.

    [:octicons-arrow-right-24: Setup](setup.md)

- :material-code-braces:{ .lg .middle } __Model Usage__

    ---

    Use pre-trained models from Hugging Face Hub or the chess-cv library in your projects.

    [:octicons-arrow-right-24: Model Usage](inference.md)

- :material-play:{ .lg .middle } __Train and Evaluate__

    ---

    Learn how to generate data, train models, and evaluate performance.

    [:octicons-arrow-right-24: Train and Evaluate](train-and-eval.md)

- :octicons-sparkle-fill-16:{ .lg .middle } __Documentation for LLM__

    ---

    Documentation in [llms.txt](https://llmstxt.org/) format. Just paste the following link into the LLM chat.

    [:octicons-arrow-right-24: llms-full.txt](llms-full.txt)

</div>
