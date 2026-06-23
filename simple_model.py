import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

print("🔄 Building fresh model...")

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Convert labels to categorical
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Build model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train_cat, epochs=5, batch_size=128, verbose=1)

# Evaluate
score = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"✅ Test accuracy: {score[1]:.4f}")

# Save
model.save('backend/mnist_model.h5')
print("✅ Saved as backend/mnist_model.h5")

# Also save with pickle
with open('backend/mnist.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Saved as backend/mnist.pkl")