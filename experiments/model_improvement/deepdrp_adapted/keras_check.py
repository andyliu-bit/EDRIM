"""Launch-card check 1 for the DeepDRP adapter: build the global network as the public code does
(drug_dose_response.py lines 275-346, commit 37f28d1), copy in the weights exported by `deepdrp.py --export-check`,
compare the outputs, and measure what the code's int32 `embedding_input` does to values and gradients.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/deepdrp_adapted/deepdrp.py --export-check check.npz
  .venvs/tfcheck/Scripts/python.exe experiments/model_improvement/deepdrp_adapted/keras_check.py check.npz
"""
import json
import math
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D, Input
from keras.models import Model

z = np.load(sys.argv[1])


def sinusoidal_embedding(x):                                         # public code lines 275-290, verbatim logic
    frequencies = tf.math.exp(tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), 32 // 2))
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    return tf.concat([tf.math.sin(angular_speeds * x), tf.math.cos(angular_speeds * x)], axis=-1)


drug_in = [Input(shape=(100,), dtype="int32", name=f"drug{k}_input") for k in (1, 2)]
embed = Embedding(input_dim=65, output_dim=128, name="smiles_embedding")
conv = [Conv1D(filters=32 * (k + 1), kernel_size=4, activation="relu", padding="valid", strides=1) for k in range(3)]
pool = GlobalMaxPooling1D()


def encode(x):                                                       # layers shared by both drugs, as in the code
    x = embed(x)
    for c in conv:
        x = c(x)
    return pool(x)


cell_in = Input(shape=(78,), name="cell_input")
cell_dense = [Dense(32, activation="relu"), Dense(64, activation="relu"), Dense(96, activation="relu")]
h = cell_in
for d in cell_dense:
    h = d(h)
dose_in = [keras.Input(shape=(1,)) for _ in range(2)]
doses = [layers.Lambda(sinusoidal_embedding, output_shape=(32,))(x) for x in dose_in]
embedding = layers.Concatenate()([encode(drug_in[0]), encode(drug_in[1]), h, *doses])
inputs = [*drug_in, cell_in, *dose_in]
model_embedding = Model(inputs=inputs, outputs=[embedding])

embed.set_weights([z["embed_weight"]])
for k, c in enumerate(conv):
    c.set_weights([z[f"conv_{k}_weight"].transpose(2, 1, 0), z[f"conv_{k}_bias"]])
for k, d in enumerate(cell_dense):
    d.set_weights([z[f"cell_{k}_weight"].T, z[f"cell_{k}_bias"]])

x = [tf.constant(z["tok1"].astype("int32")), tf.constant(z["tok2"].astype("int32")), tf.constant(z["cell"]),
     tf.constant(z["u1"].reshape(-1, 1)), tf.constant(z["u2"].reshape(-1, 1))]
emb = model_embedding(x).numpy()
diff = np.abs(emb - z["emb"])
omega_tf = (2.0 * math.pi * tf.math.exp(tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), 16))).numpy()
u = z["u1"].astype(np.float64)[:, None]
result = dict(tensorflow=tf.__version__, keras=keras.__version__, batch=int(len(emb)),
              embedding_max_abs_diff=float(diff.max()),
              drug_and_cell_part_max_abs_diff=float(diff[:, :288].max()),
              dose_part_max_abs_diff=float(diff[:, 288:].max()),
              frequency_max_rel_diff=float(np.abs(omega_tf - z["omega"]).max() / z["omega"].max()),
              # the same float64 sinusoid evaluated at the two float32 frequency vectors
              dose_diff_explained_by_frequencies=float(np.abs(np.sin(omega_tf.astype(np.float64) * u)
                                                              - np.sin(z["omega"].astype(np.float64) * u)).max()),
              embedding_values_changed_by_int32=float(np.mean(np.trunc(emb) != emb)))
same = tf.constant(z["omega"]) * x[3]                                # the code's sinusoid at our float32 frequencies
same = tf.concat([tf.math.sin(same), tf.math.cos(same)], axis=-1).numpy()
result["dose_part_max_abs_diff_same_frequencies"] = float(np.abs(same - z["emb"][:, 288:320]).max())
for dtype in ("float32", "int32"):                                   # float32 = our fix, int32 = public code
    task_in = Input(shape=(352,), dtype=dtype, name="embedding_input")
    dense3 = Dense(512, activation="relu", name="dense3_")
    dense4 = Dense(1, name="dense4_")
    model_task = Model(inputs=[task_in], outputs=[dense4(dense3(task_in))])
    dense3.set_weights([z["fc_weight"].T, z["fc_bias"]])
    dense4.set_weights([z["out_weight"].T, z["out_bias"]])
    model_pred = Model(inputs=inputs, outputs=[model_task(model_embedding(inputs))])
    watched = dict(smiles_embedding=embed.embeddings, conv1=conv[0].kernel, cell_dense1=cell_dense[0].kernel,
                   dense3=dense3.kernel)
    with tf.GradientTape() as tape:
        pred = model_pred(x, training=False)
        while isinstance(pred, (list, tuple)):                       # outputs=[...] nests the task model's list
            pred = pred[0]
        loss = tf.reduce_mean(tf.square(pred))
    grads = tape.gradient(loss, list(watched.values()))
    result[dtype] = dict(
        prediction_max_abs_diff=float(np.abs(pred.numpy().ravel() - z["pred"]).max()),
        gradient_reaches={k: bool(g is not None and float(tf.reduce_max(tf.abs(g))) > 0) for k, g in zip(watched, grads)})
print(json.dumps(result, indent=1))
