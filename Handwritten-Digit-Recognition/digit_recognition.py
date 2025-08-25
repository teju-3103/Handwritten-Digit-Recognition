import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Build a simple Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train model
print("Training the model...")
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 5. Evaluate model
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# 6. Save the model
model.save("digit_recognition_model.h5")
print("Model saved as digit_recognition_model.h5")
