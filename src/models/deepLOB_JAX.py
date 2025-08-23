import jax
from jax import jit
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from flax.training import checkpoints
from typing import Tuple
from pathlib import Path
import optax

from src.models.baseModel import BaseModel

from src.core.generalUtils import weightLocation, nameModelRun


class _DeepLOB_JAX(nn.Module):
    """
    Description:
        This under the hood for DeepLOB_JAX, it is the deepLOB architecture that forms the model
        Dont use this model durectly
    """
    input_shape: Tuple[int, int, int]
    num_lstm_units: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Convolutional block
        x = nn.Conv(features=32, kernel_size=(1, 2), strides=(1, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Conv(features=32, kernel_size=(4, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Conv(features=32, kernel_size=(4, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        x = nn.Conv(features=32, kernel_size=(1, 2), strides=(1, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Conv(features=32, kernel_size=(4, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Conv(features=32, kernel_size=(4, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        x = nn.Conv(features=32, kernel_size=(1, 10))(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Conv(features=32, kernel_size=(4, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Conv(features=32, kernel_size=(4, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        # Inception module
        branch1 = nn.Conv(64, (1, 1), padding='SAME')(x)
        branch1 = nn.leaky_relu(branch1, negative_slope=0.01)
        branch1 = nn.Conv(64, (3, 1), padding='SAME')(branch1)
        branch1 = nn.leaky_relu(branch1, negative_slope=0.01)

        branch2 = nn.Conv(64, (1, 1), padding='SAME')(x)
        branch2 = nn.leaky_relu(branch2, negative_slope=0.01)
        branch2 = nn.Conv(64, (5, 1), padding='SAME')(branch2)
        branch2 = nn.leaky_relu(branch2, negative_slope=0.01)

        branch3 = nn.max_pool(x, window_shape=(3, 1), strides=(1, 1), padding='SAME')
        branch3 = nn.Conv(64, (1, 1), padding='SAME')(branch3)
        branch3 = nn.leaky_relu(branch3, negative_slope=0.01)

        x = jnp.concatenate([branch1, branch2, branch3], axis=-1)

        # Reshape for LSTM
        B, H, W, C = x.shape
        x = x.reshape((B, H, -1))

        # Dropout across feature dimension
        if train:
            dropout_rng = self.make_rng('dropout')
            noise_shape = (B, 1, x.shape[2])
            keep_prob = 0.8
            mask = jax.random.bernoulli(dropout_rng, p=keep_prob, shape=noise_shape)
            x = jnp.where(mask, x / keep_prob, 0)

        # LSTM layer
        lstm_cell = nn.LSTMCell(features=self.num_lstm_units)
        lstm = nn.RNN(cell=lstm_cell)
        # initial_carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (B,))
        lstm_out = lstm(x)
        lstm_out = lstm_out[:, -1, :]

        # Output layer
        logits = nn.Dense(features=3)(lstm_out)
        return nn.softmax(logits)

class DeepLOB_JAX(BaseModel):
    name = "deepLOB_JAX"
    def __init__(self, input_shape: Tuple[int, int, int] = (100, 40, 1), num_lstm_units: int = 64):
        super().__init__()
        self.name = DeepLOB_JAX.name
        self.weightsFileFormat = "msgpack"
        self.input_shape = input_shape
        self.num_lstm_units = num_lstm_units
        self.model_def = _DeepLOB_JAX(self.input_shape, self.num_lstm_units)
        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1,) + input_shape)
        self.params = self.model_def.init(rng, dummy_x)
        # JIT the apply function
        self.jit_model = jit(self.model_def.apply)

    # BIG REVIEW REQUIRED
    def train(self, x, train: bool = True):
        # Provide dropout rng if training
        rngs = {"dropout": jax.random.PRNGKey(1)} if train else None
        return self.jit_model(self.params, x, rngs=rngs)

    def predict(self, x):
        # Inference: no dropout rng needed
        rngs = {"dropout": jax.random.PRNGKey(1)} 
        return self.jit_model(self.params, x, rngs=rngs)
    
    def saveWeights(self, run_id):
        # checkpoints.save_checkpoint(ckpt_dir=weightLocation(self), target=self.params)
        Path(weightLocation(self, runName=nameModelRun(runID=run_id))).write_bytes(serialization.to_bytes(self.params))

    def loss_fn(self, params, model_def, x, y):
        logits = model_def.apply(params, x, rngs={"dropout": jax.random.PRNGKey(0)})
        loss = optax.softmax_cross_entropy(logits, y).mean()
        return loss

    @jax.jit
    def train_step(self, params, model_def, x, y, optimizer, opt_state):
        grads = jax.grad(self.loss_fn)(params, model_def, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
        
if __name__ == "__main__":
    model = DeepLOB_JAX()
    model.saveWeights()