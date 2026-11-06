"""D21 check (1), runs INSIDE tensorflow/tensorflow:1.15.5-py3 with tffm==1.0.1: TFFM outputs and objective for the
weights and inputs in /io/case.npz; writes /io/tffm_out.npz. Driven by check_vs_tffm.py on the host."""
import numpy as np
import tensorflow as tf
from tffm import TFFMRegressor

case = np.load("/io/case.npz")
x, y, order, rank, reg = case["x"], case["y"], int(case["order"]), int(case["rank"]), float(case["reg"])
model = TFFMRegressor(order=order, rank=rank, n_epochs=1, optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
                      batch_size=-1, init_std=0.01, reg=reg, input_type="dense", seed=0)
model.core.set_num_features(x.shape[1])
model.core.build_graph()
model.initialize_session()
with model.core.graph.as_default():  # core.w[k] are verify_tensor_all_finite wrappers; assign the variables by name
    var = {v.op.name.split("/")[-1]: v for v in tf.global_variables()}
    for k in range(order):
        model.session.run(var[f"embedding_{k + 1}"].assign(case[f"w{k + 1}"]))
    model.session.run(var["bias"].assign(float(case["b"])))
feed = {model.core.train_x: x, model.core.train_y: y, model.core.train_w: np.ones_like(y)}
out, target = model.session.run([model.core.outputs, model.core.target], feed_dict=feed)
np.savez("/io/tffm_out.npz", outputs=np.asarray(out).ravel(), target=np.float64(target))
print("tffm outputs", np.asarray(out).ravel()[:4], "target", target)
