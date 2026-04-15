import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        epochs=15,   # 🔥 reduced (important for cloud)
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
        verbose=0   # cleaner logs
    )

    return model, history, x_test, y_test
