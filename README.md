# Image Caption Generator

A deep-learning image captioning project built with TensorFlow/Keras. The original notebook uses a pretrained **MobileNetV3Small** image feature extractor and a **two-layer Transformer decoder** to generate English captions.

## Project architecture

1. **Image encoder:** MobileNetV3Small (ImageNet-pretrained in the notebook; frozen during captioner training).
2. **Text processing:** Keras TextVectorization with a vocabulary capped at 5,000 tokens.
3. **Caption decoder:** Two Transformer decoder layers with causal self-attention, cross-attention and feed-forward blocks.
4. **Generation:** Autoregressive greedy decoding, with a maximum sequence length of 50 tokens.

## Dataset and training

The notebook selects the Flickr8k dataset. It preprocesses images to 224 × 224 RGB, extracts image features, tokenizes captions and trains with Adam (learning rate 1e-4), masked sparse categorical cross-entropy and masked accuracy. Early stopping uses patience 5 and restores the best weights.

The notebook's saved output includes training logs through epoch 35, where the displayed validation masked accuracy is approximately 0.3851. This is a notebook training-log observation, not an independently reproduced evaluation result.

## Run the original notebook

[Open the Google Colab notebook](https://colab.research.google.com/drive/1OL8Wmik5nfes1NrF2gzabtl-wthCDk3H?usp=sharing)

Run the notebook through model training before exporting the trained artifacts.

## Export the trained model

Add and run this cell **after training and after the model has been built** in the notebook:

```python
import json
import os
import shutil
import zipfile
import numpy as np

export_dir = "/content/captioner_export"
os.makedirs(export_dir, exist_ok=True)

# Ensure the MobileNet encoder is built and included in the checkpoint.
_ = mobilenet(tf.zeros((1, 224, 224, 3), dtype=tf.float32))

# Build the model graph with image-feature input and a start token.
start_tokens = word_to_index([["[START]"]])
_ = model((tf.zeros((1, 7, 7, 576)), start_tokens), training=False)

model.save_weights(os.path.join(export_dir, "captioner.weights.h5"))

with open(os.path.join(export_dir, "vocabulary.json"), "w") as f:
    json.dump(tokenizer.get_vocabulary(), f)

np.save(os.path.join(export_dir, "output_bias.npy"), output_layer.bias)

archive = shutil.make_archive("/content/captioner_export", "zip", export_dir)

from google.colab import files
files.download(archive)
```

The exported ZIP should contain exactly the three files listed above. Extract them into the app's `artifacts/` directory. Keep these model binaries out of Git if they are large; use a model-artifact host or the deployment platform's supported storage for production.

## Run the Streamlit application locally

After the model files are in `artifacts/`:

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

The application is designed for Streamlit Community Cloud. Push the source code to GitHub, connect the repository in Streamlit Community Cloud, set the app entry point to `app.py`, and provide the three model artifacts through a suitable model-artifact hosting method. Do not publish private data or credentials.

**Deployment status:** Application source scaffold is prepared. A live inference deployment is pending export and verification of the trained model artifacts.

## Repository structure

```text
Image-Caption-Generator/
├── app.py
├── requirements.txt
├── README.md
├── Google Colab Link
└── artifacts/
    └── README.md
```

## Limitations and responsible use

The original notebook notes that Flickr8k is a relatively small dataset and that captions for images outside its training distribution may be inaccurate. Generated captions should be treated as model predictions, not verified facts.

## Author

**Preeti Dagar** · [GitHub](https://github.com/PreetiDagar) · [Portfolio](https://meek-mermaid-723af2.netlify.app/) · [Email](mailto:dagarp72@gmail.com)
