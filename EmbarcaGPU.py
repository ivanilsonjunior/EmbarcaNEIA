import tensorflow as tf

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Create dummy datas
import numpy as np
x_train = np.random.rand(100, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 100).astype(np.int32)

# Train the model, TensorFlow will automatically place operations on GPU if available
# and configured correctly.
print("Starting training...")
model.fit(x_train, y_train, epochs=1)
print("Training complete.")