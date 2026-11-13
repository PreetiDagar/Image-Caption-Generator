# Model artifacts

This folder is intentionally kept free of trained model binaries in Git.

After training the model in the original Google Colab notebook, export these three files:

- `captioner.weights.h5` — trained Keras model weights
- `vocabulary.json` — TextVectorization vocabulary
- `output_bias.npy` — output-layer bias learned from caption token frequencies

Place the three files in this folder before running or deploying the Streamlit app.

See the root README for the exact export cell. Do not commit API keys, private datasets, or user images.
