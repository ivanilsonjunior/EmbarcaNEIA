import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

# ============================================================
# 1. LIMPEZA DE IMAGENS CORROMPIDAS (MESMO ESQUEMA QUE VOCÊ JÁ USAVA)
# ============================================================
num_skipped = 0
for folder_name in ("Cuscuz", "Tapioca"):
    folder_path = os.path.join("Imagens/Treinamento", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)  # apaga imagem corrompida

print(f"Deleted {num_skipped} images.")

# ============================================================
# 2. GERAR DATASETS DE TREINO/VALIDAÇÃO
# ============================================================
image_size = (180, 180)
batch_size = 16  # menor, melhor para dataset pequeno

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "Imagens/Treinamento",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
print("Classes:", class_names)  # deve mostrar ['Cuscuz', 'Tapioca']

# ============================================================
# 3. VISUALIZA ALGUMAS IMAGENS DO TREINO
# ============================================================
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# 4. DATA AUGMENTATION (PREVIEW + DENTRO DO MODELO)
# ============================================================

# Instância de augmentation só para preview (opcional)
augmentation_preview = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="augmentation_preview",
)

# Preview de imagens aumentadas
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    augmented_images = augmentation_preview(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[i]).astype("uint8"))
        plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# 5. OTIMIZAÇÃO DO PIPELINE (SEM .repeat, SEM steps_per_epoch)
# ============================================================
train_ds = train_ds.shuffle(100).cache().prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf_data.AUTOTUNE)

# ============================================================
# 6. MODELO CNN PEQUENO (MUITO MAIS ADEQUADO PARA 140 IMAGENS)
# ============================================================
def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Data augmentation dentro do modelo
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # Blocos convolucionais bem menores
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # Binário → 1 logit + BinaryCrossentropy(from_logits=True)
    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)


tf.keras.backend.clear_session()
model = make_model(input_shape=image_size + (3,))
model.summary()

# ============================================================
# 7. COMPILAÇÃO E CALLBACKS
# ============================================================
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    # run_eagerly=True  # use isso só se precisar debugar
)

epochs = 300

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(log_dir="logs"),
]

# ============================================================
# 8. TREINAMENTO
# ============================================================
history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    verbose=1,
)

model.save("Nordestino.keras")

# ============================================================
# 9. FUNÇÃO DE INFERÊNCIA PARA TESTAR UMA IMAGEM
# ============================================================
def classificar_imagem(caminho, model, image_size):
    img = keras.utils.load_img(caminho, target_size=image_size)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # batch axis

    logits = model.predict(img_array)[0][0]
    score = float(tf.sigmoid(logits))  # probabilidade da classe "1" (Tapioca)

    print(f"Logit bruto: {logits:.4f}")
    print(f"Prob. Tapioca: {100 * score:.2f}%")
    print(f"Prob. Cuscuz : {100 * (1 - score):.2f}%")

    if score > 0.5:
        print("→ Modelo acha que é **Tapioca**")
    else:
        print("→ Modelo acha que é **Cuscuz**")
    print("-" * 50)


# Testes
classificar_imagem("Imagens/Teste/Cuscuz.jpg", model, image_size)
classificar_imagem("Imagens/Teste/Tapioca.jpg", model, image_size)
