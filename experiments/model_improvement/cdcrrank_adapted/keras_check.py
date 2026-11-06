"""Launch-card check 1 for the CDCR-Rank adapter: build the model as the public code does (drug_dose_urank.py
lines 257-461, commit 9240ec8; copied below, logic verbatim), copy in the weights exported by
`cdcrrank.py --export-check`, compare the outputs, and check three points of the code: what mask_zero does, whether
its uRank term has a gradient, and where its gain 2^s - 1 overflows.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/cdcrrank_adapted/cdcrrank.py --export-check check.npz
  .venvs/tfcheck/Scripts/python.exe experiments/model_improvement/cdcrrank_adapted/keras_check.py check.npz
"""
import json
import sys
import warnings

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model


class SinusoidalDoseEncoder(layers.Layer):                        # public code lines 257-293
    def __init__(self, embedding_dims=64, **kwargs):
        super(SinusoidalDoseEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims

    def call(self, inputs):
        r = tf.expand_dims(inputs, -1)
        i = tf.range(self.embedding_dims, dtype=tf.float32)
        exponents = -4.0 * i / self.embedding_dims
        factors = tf.pow(10.0, exponents)
        argument = r * tf.reshape(factors, [1, 1, -1])
        sin_mask = tf.cast(i % 2 == 0, tf.float32)
        cos_mask = tf.cast(i % 2 == 1, tf.float32)
        sin_component = tf.reshape(sin_mask, [1, 1, -1]) * tf.math.sin(argument)
        cos_component = tf.reshape(cos_mask, [1, 1, -1]) * tf.math.cos(argument)
        encoding = sin_component + cos_component
        encoding = tf.squeeze(encoding, axis=1)
        return encoding


class URankLoss(keras.losses.Loss):                               # public code lines 296-347
    def __init__(self, name='urank_loss', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, -1)
        if len(y_pred.shape) == 1:
            y_pred = tf.expand_dims(y_pred, -1)
        n = tf.shape(y_true)[0]
        G = tf.math.pow(2.0, y_true) - 1.0
        e = tf.ones_like(y_true)
        S = y_true
        S_T = tf.transpose(S)
        e_T = tf.transpose(e)
        term1 = tf.matmul(S, e_T)
        term2 = tf.matmul(e, S_T)
        P = term1 - term2
        M = tf.cast(tf.greater(P, 0), tf.float32)
        O = y_pred
        T = M * tf.math.exp(O)
        L = tf.math.log(1.0 + T / tf.math.exp(O))
        L = L * G
        loss = tf.reduce_sum(L) / tf.cast(n - 1, tf.float32)
        return loss


class HybridLoss(keras.losses.Loss):                              # public code lines 350-369
    def __init__(self, lambda_param=0.5, name='hybrid_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_param = lambda_param
        self.mse = keras.losses.MeanSquaredError()
        self.urank = URankLoss()

    def call(self, y_true, y_pred):
        mse_loss = self.mse(y_true, y_pred)
        urank_loss = self.urank(y_true, y_pred)
        return self.lambda_param * mse_loss + (1 - self.lambda_param) * urank_loss


def build_cdcr_rank_model(MAX_SMI_LEN=100, n_cell_features=78, drug_embed_dim=128, dose_embed_dim=64):  # 376-461
    drug1_input = layers.Input(shape=(MAX_SMI_LEN,), name='drug1_input', dtype='float32')
    drug2_input = layers.Input(shape=(MAX_SMI_LEN,), name='drug2_input', dtype='float32')
    dose1_input = layers.Input(shape=(1,), name='dose1_input')
    dose2_input = layers.Input(shape=(1,), name='dose2_input')
    cell_line_input = layers.Input(shape=(n_cell_features,), name='cell_line_input')
    drug1_int = layers.Lambda(lambda x: tf.cast(x, tf.int32))(drug1_input)
    drug2_int = layers.Lambda(lambda x: tf.cast(x, tf.int32))(drug2_input)
    smiles_embedding = layers.Embedding(input_dim=65, output_dim=drug_embed_dim, input_length=MAX_SMI_LEN,
                                        mask_zero=True, name='smiles_embedding')

    def build_drug_cnn():
        return keras.Sequential([
            layers.Conv1D(32, 4, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 4, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 4, activation='relu', padding='same'),
            layers.GlobalMaxPooling1D()
        ], name='drug_cnn')

    drug_cnn = build_drug_cnn()
    drug1_embedded = smiles_embedding(drug1_int)
    drug2_embedded = smiles_embedding(drug2_int)
    drug1_encoded = drug_cnn(drug1_embedded)
    drug2_encoded = drug_cnn(drug2_embedded)
    dose_encoder = SinusoidalDoseEncoder(embedding_dims=dose_embed_dim)
    dose1_encoded = dose_encoder(dose1_input)
    dose2_encoded = dose_encoder(dose2_input)
    cell_encoded = layers.Dense(128, activation='relu')(cell_line_input)
    cell_encoded = layers.Dense(256, activation='relu')(cell_encoded)
    cell_encoded = layers.Dense(128, activation='relu')(cell_encoded)
    concatenated = layers.Concatenate()([drug1_encoded, drug2_encoded, dose1_encoded, dose2_encoded, cell_encoded])
    x = layers.Dense(256, activation='relu')(concatenated)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    synergy_output = layers.Dense(1, name='synergy_output')(x)
    return Model(inputs=[drug1_input, drug2_input, dose1_input, dose2_input, cell_line_input], outputs=synergy_output,
                 name='CDCR_Rank')


z = np.load(sys.argv[1])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = build_cdcr_rank_model(n_cell_features=z["cell"].shape[1])
    model.get_layer("smiles_embedding").set_weights([z["embed_weight"]])
    convs = [l for l in model.get_layer("drug_cnn").layers if isinstance(l, layers.Conv1D)]
    for k, c in enumerate(convs):
        c.set_weights([z[f"conv_{k}_weight"].transpose(2, 1, 0), z[f"conv_{k}_bias"]])
    dense = [l for l in model.layers if isinstance(l, layers.Dense)]       # cell MLP (3), then head (4)
    ours = [f"cell_{k}" for k in range(3)] + [f"head_{k}" for k in range(4)]
    for d, name in zip(dense, ours, strict=True):
        d.set_weights([z[f"{name}_weight"].T, z[f"{name}_bias"]])
    concat_layer = next(l for l in model.layers if isinstance(l, layers.Concatenate))
    probe = Model(model.inputs, [concat_layer.output, model.output])
    x = [tf.constant(z["tok1"].astype("float32")), tf.constant(z["tok2"].astype("float32")),
         tf.constant(z["u1"].reshape(-1, 1)), tf.constant(z["u2"].reshape(-1, 1)), tf.constant(z["cell"])]
    concat, pred = (t.numpy() for t in probe(x, training=False))
    emb_mask = model.get_layer("smiles_embedding").compute_mask(tf.cast(x[0], tf.int32))
diff = np.abs(concat - z["concat"])
parts = dict(drug1=(0, 128), drug2=(128, 256), dose1=(256, 320), dose2=(320, 384), cell=(384, 512))
result = dict(tensorflow=tf.__version__, keras=keras.__version__, batch=int(len(pred)),
              max_abs_diff={p: float(diff[:, a:b].max()) for p, (a, b) in parts.items()}
              | {"prediction": float(np.abs(pred.ravel() - z["pred"]).max())},
              padding_tokens_share=float(np.mean(z["tok1"] == 0)),
              mask=dict(embedding_returns_mask=emb_mask is not None,
                        conv1d_supports_masking=bool(convs[0].supports_masking),
                        warnings=sorted({str(w.message)[:240] for w in caught if "mask" in str(w.message).lower()})))
y_true = tf.constant(z["list_y"].reshape(-1, 1).astype("float32"))           # one 15-cell validation list, percent
result["code_urank"] = dict(list_cells=int(len(z["list_y"])), list_max_percent=float(z["list_y"].max()))
for label, mean, sd in (("predictions_percent_scale", 50.0, 20.0), ("predictions_below_44", 0.5, 0.2)):
    o = tf.Variable(np.random.default_rng(0).normal(mean, sd, (len(z["list_y"]), 1)).astype("float32"))
    with tf.GradientTape(persistent=True) as tape:
        lu, lh, lm = URankLoss()(y_true, o), HybridLoss(0.5)(y_true, o), keras.losses.MeanSquaredError()(y_true, o)
    gu, gh, gm = (tape.gradient(v, o).numpy() for v in (lu, lh, lm))
    o2 = tf.Variable(o.numpy() + 1.0)                                       # value unchanged when O moves
    result["code_urank"][label] = dict(
        max_prediction=float(o.numpy().max()), value=float(lu), value_after_shift=float(URankLoss()(y_true, o2)),
        grad_max_abs=float(np.abs(gu).max()), hybrid_grad_minus_half_mse_grad_max_abs=float(np.abs(gh - 0.5 * gm).max()),
        mse_grad_max_abs=float(np.abs(gm).max()))
result["code_gain_2s_minus_1"] = {f"{v:g}": float(tf.pow(2.0, v) - 1.0) for v in (-50.0, 100.0, 128.0, 129.0, 150.0)}
print(json.dumps(result, indent=1))
