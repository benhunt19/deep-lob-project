import jax
from src.models.deepLOB_JAX import DeepLOB_JAX
import jax.numpy as jnp

if __name__ == "__main__":
    model = DeepLOB_JAX(input_shape=(100, 40, 1), num_lstm_units=64)
    vals = jnp.ones((1000, 100, 40, 1))
    vals[:,:,:,]
    model.train(x=vals)
    ret = model.predict(jnp.ones((5, 100, 40, 1)))
    print(ret)