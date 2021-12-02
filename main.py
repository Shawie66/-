import numpy as np
import random
import cv2
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import uuid

# set up path
pos_path = os.path.join('data', 'positive')
neg_path = os.path.join('data', 'negative')
anchor_path = os.path.join('data', 'anchor')
# os.makedirs(pos_path)
# os.makedirs(neg_path)
# os.makedirs(anchor_path)

# Move lfw images to the following repository data/negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        ex_path = os.path.join('lfw', directory, file)
        new_path = os.path.join(neg_path, file)
        os.replace(ex_path, new_path)

# Establish a connection to the camera
# capture = cv2.VideoCapture(0)
# while capture.isOpened():
#     ret, frame = capture.read()
#
#     # cut down frame
#     frame = frame[120:120 + 250, 200:200 + 250, :]
#
#     # collect anchors
#     if cv2.waitKey(1) & 0XFF == ord('a'):
#         # create the unique file path
#         imgname = os.path.join(anchor_path, '{}.jpg'.format(uuid.uuid1()))
#
#         # write out anchor image
#         cv2.imwrite(imgname, frame)
#     # collect positives
#     if cv2.waitKey(1) & 0XFF == ord('p'):
#         # create the unique file path
#         imgname = os.path.join(pos_path, '{}.jpg'.format(uuid.uuid1()))
#
#         # write out positive image
#         cv2.imwrite(imgname, frame)
#
#     # show image back to screen
#     cv2.imshow('Image Collection', frame)
#
#     # Breaking with pressing q
#     if cv2.waitKey(1) & 0XFF == ord('q'):
#         break
#
# capture.release()
# cv2.destroyAllWindows()
# plt.imshow(frame[120:120 + 250, 200:200 + 250, :])

# Get image directories
anchor = tf.data.Dataset.list_files(anchor_path + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(pos_path + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(neg_path + '\*.jpg').take(300)


# Preprocessing imgage - Scale and Resize
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)

    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Resizing the image to be 100*100*3
    img = tf.image.resize(img, (105, 105))
    img = img / 255.0
    return img


# img = preprocess('data\\anchor\\2569ff11-51ea-11ec-b937-34c93d55aec8.jpg')
# plt.imshow(img)
# plt.show()

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)
samples = data.as_numpy_iterator()
example = samples.next()


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# Build Dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training partition
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# Build Embedding Layer
def make_embedding():
    inp = Input(shape=(105, 105, 3), name='input_image')
    conv1 = Conv2D(64, (10, 10), strides=1, activation='relu')(inp)
    maxpool1 = MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = Conv2D(128, (7, 7), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (4, 4), activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D((2, 2), padding='same')(conv3)

    conv4 = Conv2D(256, (4, 4), activation='relu')(maxpool3)
    f1 = Flatten()(conv4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()


class L1Dis(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_emdbedding, validation_embedding):
        return tf.math.abs(input_emdbedding - validation_embedding)


l1 = L1Dis()


def make_siamese_model():
    # anchor image input in the network
    input_image = Input(name='input_img', shape=(105, 105, 3))

    # validation image input in the network
    validation_image = Input(name='validation_img', shape=(105, 105, 3))

    # combine siamese distance components
    siamese_layer = L1Dis()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()

# Setup Loss and Optimizer
binary_cross_entropy = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

# Directory to save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]

        # Get label
        y = batch[2]

        y_hat = siamese_model(X, training=True)
        loss = binary_cross_entropy(y, y_hat)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss


def train(data, epochs):
    # Loop through epochs
    for epoch in range(1, epochs + 1):
        print('\n Epoch {} / {}'.format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx + 1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


epochs = 50
# train(train_data, epochs)

# Evaluate Model
from tensorflow.keras.metrics import Precision, Recall

# Get a batch of data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# Make Predictions
y_hat = siamese_model.predict([test_input, test_val])
te = [1 if prediction > 0.5 else 0 for prediction in y_hat]
# print(te)

recall = Recall()
recall.update_state(y_true, y_hat)
# print(y_true)
# print(recall.result().numpy())

# Visualizing the truth and pred results
# for i in range(len(y_true)):
#     plt.figure(figsize=(18, 8))
#     plt.subplot(1, 2, 1)
#     plt.imshow(test_input[i])
#     plt.subplot(1, 2, 2)
#     plt.imshow(test_val[i])
#     plt.show()

# Save weights
# siamese_model.save('siamesemodel.h5')

# Reload Model
model = tf.keras.models.load_model('siamesemodel.h5',
                                   custom_objects={'L1Dis': L1Dis, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})


# print(model.predict([test_input, test_val]))


# Real-Time Test
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_image = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_image = preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_image, validation_image], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


# OpenCV Real-Time Verification
cap = cv2.VideoCapture(0)
while cap.isOpened():

    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]
    cv2.imshow('Verification', frame)

    # Verification Trigger
    if cv2.waitKey(5) & 0XFF == ord('v'):
        # Save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)

        # Run verification
        results, verified = verify(model, 0.9, 0.9)
        print(verified)

    if cv2.waitKey(5) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(np.sum(np.squeeze(results) > 0.9))
