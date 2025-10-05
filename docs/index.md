<figure markdown="span">
  ![Chess CV](assets/model.svg){ width="256" }
  <figcaption>Lightweight CNN for chess piece classification</figcaption>
</figure>

---

!!! info "Project Status"

    This project is actively developed. The current version provides a complete pipeline for training a lightweight CNN to classify chess pieces from 32×32px square images.

Chess CV is a machine learning project that trains a lightweight CNN (156k parameters) from scratch to classify chess pieces from synthetically generated chessboard square images. The project combines 55 board styles with 64 piece sets from chess.com and lichess to generate a diverse training dataset of approximately 93,000 images. By rendering pieces onto different board backgrounds and extracting individual squares, the model learns robust piece recognition across various visual styles.

<div class="grid cards" markdown>

- :material-cog:{ .lg .middle } __Setup__

    ---

    Installation guide covering dependencies, environment setup, and data preparation.

    [:octicons-arrow-right-24: Setup](setup.md)

- :material-play:{ .lg .middle } __Usage__

    ---

    Learn how to generate data, train models, and evaluate performance.

    [:octicons-arrow-right-24: Usage](usage.md)

- :material-code-braces:{ .lg .middle } __Model Usage__

    ---

    Use pre-trained models from Hugging Face Hub or the chess-cv library in your projects.

    [:octicons-arrow-right-24: Model Usage](inference.md)

- :octicons-sparkle-fill-16:{ .lg .middle } __Documentation for LLM__

    ---

    Documentation in [llms.txt](https://llmstxt.org/) format. Just paste the following link (or its content) into the LLM chat.

    [:octicons-arrow-right-24: llms-full.txt](llms-full.txt)

</div>
