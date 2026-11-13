import json
from pathlib import Path

import einops
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Image Caption Studio | Preeti Dagar",
    page_icon="🖼️",
    layout="wide",
)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ARTIFACTS / "captioner.weights.h5"
VOCAB = ARTIFACTS / "vocabulary.json"
BIAS = ARTIFACTS / "output_bias.npy"
IMAGE_SHAPE = (224, 224, 3)


class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(max_length, depth)
        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size, depth, mask_zero=True
        )
        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq = self.token_embedding(seq)
        positions = tf.range(tf.shape(seq)[1])[tf.newaxis, :]
        return self.add([seq, self.pos_embedding(positions)])


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attended = self.mha(query=x, value=x, use_causal_mask=True)
        return self.layernorm(self.add([x, attended]))


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.last_attention_scores = None

    def call(self, x, y):
        attended, scores = self.mha(
            query=x, value=y, return_attention_scores=True
        )
        self.last_attention_scores = scores
        return self.layernorm(self.add([x, attended]))


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(2 * units, activation="relu"),
            tf.keras.layers.Dense(units),
            tf.keras.layers.Dropout(dropout_rate),
        ])
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        return self.layernorm(x + self.seq(x, training=training))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.ff = FeedForward(units, dropout_rate)

    def call(self, inputs, training=False):
        image_features, text_sequence = inputs
        text_sequence = self.self_attention(text_sequence)
        text_sequence = self.cross_attention(text_sequence, image_features)
        return self.ff(text_sequence, training=training)


class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, vocab_size):
        super().__init__()
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.bias = None

    def call(self, x):
        return self.dense(x) + self.bias


class Captioner(tf.keras.Model):
    def __init__(self, vocabulary, feature_extractor, output_layer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="", vocabulary=vocabulary
        )
        self.seq_embedding = SeqEmbedding(
            vocab_size=len(vocabulary), max_length=50, depth=256
        )
        self.decoder_layers = [
            DecoderLayer(256, num_heads=2, dropout_rate=0.5)
            for _ in range(2)
        ]
        self.output_layer = output_layer

    def call(self, inputs, training=False):
        image_features, tokens = inputs
        image_features = einops.rearrange(
            image_features, "b h w c -> b (h w) c"
        )
        text_sequence = self.seq_embedding(tokens)
        for layer in self.decoder_layers:
            text_sequence = layer(
                (image_features, text_sequence), training=training
            )
        return self.output_layer(text_sequence)


@st.cache_resource(show_spinner="Loading the trained captioning model...")
def load_captioner():
    required = [WEIGHTS, VOCAB, BIAS]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model files are not available: "
            + ", ".join(missing)
            + ". Export the trained model from the Colab notebook first."
        )

    vocabulary = json.loads(VOCAB.read_text(encoding="utf-8"))
    output_bias = np.load(BIAS)

    encoder = tf.keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        include_preprocessing=True,
        weights=None,
    )
    encoder.trainable = False

    output_layer = TokenOutput(len(vocabulary))
    output_layer.bias = output_bias

    captioner = Captioner(vocabulary, encoder, output_layer)

    # Build all model layers before restoring the notebook's saved weights.
    dummy_image = tf.zeros((1, 7, 7, 576), dtype=tf.float32)
    start_id = int(captioner.word_to_index("[START]").numpy())
    dummy_tokens = tf.constant([[start_id]], dtype=tf.int64)
    captioner((dummy_image, dummy_tokens), training=False)
    captioner.load_weights(str(WEIGHTS))

    return captioner, encoder, vocabulary, captioner.word_to_index


def generate_caption(image, captioner, encoder, vocabulary, word_to_index):
    image = image.convert("RGB").resize((224, 224))
    pixels = np.asarray(image, dtype=np.float32)[None, ...]
    image_tensor = tf.convert_to_tensor(pixels)

    image_features = encoder(image_tensor, training=False)
    start_id = int(word_to_index("[START]").numpy())
    end_id = int(word_to_index("[END]").numpy())
    tokens = tf.constant([[start_id]], dtype=tf.int64)

    generated = []
    for _ in range(50):
        logits = captioner((image_features, tokens), training=False)
        next_id = int(tf.argmax(logits[:, -1, :], axis=-1)[0].numpy())
        if next_id == end_id:
            break
        if next_id <= 1 or next_id >= len(vocabulary):
            break
        word = vocabulary[next_id]
        if word not in ("", "[UNK]", "[START]", "[END]"):
            generated.append(word)
        tokens = tf.concat(
            [tokens, tf.constant([[next_id]], dtype=tf.int64)], axis=1
        )

    return " ".join(generated).strip()


st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #edf2ff; }
    .block-container { max-width: 1120px; padding-top: 2.5rem; }
    .hero {
        padding: 2rem; border: 1px solid #293653; border-radius: 22px;
        background: linear-gradient(135deg, #151f38, #111827);
        margin-bottom: 1.5rem;
    }
    .eyebrow { color: #9caeff; letter-spacing: .13em; font-size: .78rem; }
    .hero h1 { color: #f8fafc; margin: .5rem 0; font-size: 2.5rem; }
    .hero p { color: #c2cce0; font-size: 1.05rem; }
    .caption-card {
        padding: 1.2rem; border-radius: 16px; background: #151f38;
        border: 1px solid #334466; color: #f8fafc; font-size: 1.15rem;
    }
    </style>
    <div class="hero">
      <div class="eyebrow">COMPUTER VISION · DEEP LEARNING</div>
      <h1>Image Caption Studio</h1>
      <p>Upload an image and generate a natural-language description
      using a MobileNetV3 image encoder and Transformer decoder.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("01 · Upload an image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear image for best results.",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

with right:
    st.subheader("02 · Generate a caption")
    st.write(
        "The model analyses visual features and predicts a caption "
        "one token at a time."
    )
    generate = st.button(
        "Generate image caption",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )
    if generate and uploaded:
        try:
            captioner, encoder, vocabulary, word_to_index = load_captioner()
            with st.spinner("Analysing image and generating caption..."):
                result = generate_caption(
                    image, captioner, encoder, vocabulary, word_to_index
                )
            if result:
                st.markdown("**Generated caption**")
                st.markdown(
                    f'<div class="caption-card">{result}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("The model did not generate a caption for this image.")
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info(
                "The notebook's trained weights, vocabulary and output bias "
                "must be exported and placed in the artifacts folder."
            )
        except Exception as exc:
            st.exception(exc)

st.divider()
st.caption(
    "Built by Preeti Dagar · TensorFlow · MobileNetV3Small · Transformer Decoder"
)
