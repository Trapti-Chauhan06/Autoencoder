import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], -1))  
x_test = x_test.reshape((x_test.shape[0], -1))      

def blur_images(images, sigma=2.0):
    return np.array([gaussian_filter(img.reshape(28, 28), sigma=sigma).reshape(784) for img in images])

x_train_blurred = blur_images(x_train, sigma=2.0)
x_test_blurred = blur_images(x_test, sigma=2.0)

input_dim = x_train.shape[1]  
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(
    x_train_blurred, x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_blurred, x_test)
)

decoded_imgs = autoencoder.predict(x_test_blurred)
n = 10
plt.figure(figsize=(20, 6))

for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_blurred[i].reshape(28, 28), cmap='gray')
    plt.title("Blurred")
    plt.axis('off')
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()
