import tensorflow as tf
import numpy as np

# Dummy data (replace with your dataset, e.g., order book features)
np.random.seed(42)
x_train = np.random.rand(1000, 10).astype('float32')  # 10 features
y_train = np.random.randint(0, 2, (1000, 1)).astype('float32')  # Binary classification

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,), name="input_layer"),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=32)

# Save as .keras file
model.save('lob_model.keras')

# Convert to SavedModel
model.export('lob_model_saved')