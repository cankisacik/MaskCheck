# MaskCheck: Real-Time Face Mask Detection Using CNN-Transformer Hybrid

# library importing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D,
                                     Multiply, Reshape, Activation, Concatenate, BatchNormalization, Add,
                                     LayerNormalization, concatenate)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import cv2
import os
import pickle
from google.colab import files
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# data importing
import zipfile

zip_path = '/content/archive.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("dataset")

data_path = 'dataset/Face Mask Dataset'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# attention layers and custom modules
def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])

class ReduceMean(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

class ReduceMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)

def cbam_block(input_tensor):
    channel = squeeze_excite_block(input_tensor)
    avg_pool = ReduceMean()(channel)
    max_pool = ReduceMax()(channel)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    return Multiply()([channel, cbam_feature])

class ExpandDimsTwice(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.expand_dims(inputs, 1)
        return tf.expand_dims(x, 1)

def eca_block(input_tensor, k_size=3):
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = ExpandDimsTwice()(avg_pool)
    conv = Conv2D(1, (1, k_size), padding='same', activation='sigmoid')(avg_pool)
    return Multiply()([input_tensor, conv])

def non_local_block(x):
    _, h, w, c = x.shape
    x_reshaped = tf.keras.layers.Reshape((h * w, c))(x)
    f = Dense(c // 8)(x_reshaped)
    g = Dense(c // 8)(x_reshaped)
    h_matrix = Dense(c)(x_reshaped)
    s = tf.keras.layers.Dot(axes=(2, 2))([g, f])
    beta = Activation('softmax')(s)
    o = tf.keras.layers.Dot(axes=(2, 1))([beta, h_matrix])
    o = Reshape((h, w, c))(o)
    return Add()([x, o])

def multi_head_self_attention_block(x, num_heads=4):
    d_model = x.shape[-1]
    h, w = x.shape[1], x.shape[2]
    x_flat = Reshape((h * w, d_model))(x)
    mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    attn_output = mha(x_flat, x_flat)
    x_out = LayerNormalization()(x_flat + attn_output)
    return Reshape((h, w, d_model))(x_out)

def custom_conv_block(x, filters):
    x1 = Conv2D(filters, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(filters, (1, 1), padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    return Add()([x1, x2])

def build_final_maskcheck_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    cbam = cbam_block(x)
    eca = eca_block(cbam)
    se = squeeze_excite_block(eca)
    conv_features = custom_conv_block(se, 128)
    nonlocal_features = non_local_block(se)
    transformer_features = multi_head_self_attention_block(se)
    fused = concatenate([conv_features, nonlocal_features, transformer_features], axis=-1)
    x = GlobalAveragePooling2D()(fused)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=outputs)

# computing class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)


# training the model
model = build_final_maskcheck_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('maskcheck_best_model.keras', save_best_only=True)
    ],
    verbose=1
)

val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

model.save('maskcheck_final_model.keras')

with open('maskcheck_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# plotting
with open('maskcheck_training_history.pkl', 'rb') as f:
    saved_history = pickle.load(f)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(saved_history['accuracy'], label='Train Acc')
plt.plot(saved_history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(saved_history['loss'], label='Train Loss')
plt.plot(saved_history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# gradcam
def get_img_array(img_path, size):
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1', pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path='cam.jpg', alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Grad-CAM Visualization')
    plt.show()

# loading and applying gradcam
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img_array = get_img_array(img_path, (224, 224))
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1')
save_and_display_gradcam(img_path, heatmap)

# mask status

class_labels = {
    0: 'Mask',
    1: 'No Mask',
    2: 'Incorrect Mask'
}

def predict_mask_status(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    label = class_labels[predicted_class]
    confidence = predictions[0][predicted_class] * 100

    # Görselleştir
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cv)
    plt.title(f"Prediction: {label}")
    plt.axis('off')
    plt.show()

    print(f"Prediction: {label} ({confidence:.2f}% confidence)")

    return label, confidence

label, conf = predict_mask_status(img_path)
print("Predicted class:", label)
print("Confidence:", round(conf, 2), "%")

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_true = []
y_pred = []

for i in range(len(val_generator)):
    x_batch, y_batch = val_generator[i]
    preds = model.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

labels = list(val_generator.class_indices.keys())

# confusion mateix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# feature extraction
feature_extractor = tf.keras.models.Model(
    inputs=model.input,
    outputs=model.get_layer(index=-4).output  # GAP sonrası dense'den önceki katman
)

features = []
labels = []

# subset sample
for i in tqdm(range(70)):
    img, label = val_generator[i]
    feat = feature_extractor.predict(img)
    features.append(feat[0])
    labels.append(np.argmax(label[0]))

features = np.array(features)
labels = np.array(labels)

# reduction with pca
pca = PCA(n_components=50)
pca_result = pca.fit_transform(features)

# t-sne 2d
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(pca_result)

# plotting
plt.figure(figsize=(10, 6))
for class_idx in np.unique(labels):
    plt.scatter(tsne_result[labels == class_idx, 0],
                tsne_result[labels == class_idx, 1],
                label=f"Class {class_idx}", alpha=0.7)

plt.legend()
plt.title("Feature Embedding Visualization using t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("tsne_embedding.png")
plt.show()

class_labels = ['Mask', 'No Mask', 'Incorrect Mask']

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2])

x = np.arange(len(class_labels))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1 Score')

plt.xticks(x, class_labels)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Evaluation Metrics per Class')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

import time
start = time.time()
model.predict(img_array)
end = time.time()
print(f"Inference time: {(end-start)*1000:.2f} ms")

model.summary()
