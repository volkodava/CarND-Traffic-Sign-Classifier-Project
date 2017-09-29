import csv
import glob
import os
import pickle
import sys
import warnings
import zipfile
from functools import partial
from os.path import basename
from urllib.request import urlretrieve

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from tqdm import tqdm


def download_file(url, filepath):
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d / %d" % (
                percent, len(str(totalsize)), readsofar, totalsize)
            sys.stderr.write(s)
            if readsofar >= totalsize:
                sys.stderr.write("\n")
        else:
            sys.stderr.write("read %d\n" % (readsofar,))

    print("Download %s to %s" % (url, filepath))
    if not os.path.exists(filepath):
        urlretrieve(url, filepath, reporthook)
    print("Done")


def unzip(filepath, output_dir):
    with zipfile.ZipFile(filepath) as z:
        print("Unzip %s to %s" % (filepath, output_dir))
        for filename in z.namelist():
            if not os.path.exists(os.path.join(output_dir, filename)):
                print("Extract %s" % filename)
                z.extract(filename, output_dir)
                print("Done")

        z.extractall(output_dir)
        print("Done")


def get_input_files(dataset_url, output_directory):
    res = {}
    dataset_filename = dataset_url.rsplit('/', 1)[-1]
    filepath = os.path.join(output_directory, dataset_filename)

    download_file(dataset_url, filepath)
    unzip(filepath, output_directory)

    for input_file in glob.glob(os.path.join(output_directory, '*.p')):
        filename = ".".join(basename(input_file).split(".")[:-1])
        res[filename] = input_file

    return res


def read_csv(filepath, skip_header=True):
    res = {}
    if not os.path.exists(filepath):
        return res

    with open(filepath, mode='r') as infile:
        reader = csv.reader(infile)
        is_header = True
        for row in reader:
            if is_header and skip_header:
                is_header = False
                continue
            dkey, dvalue = row
            res[int(dkey)] = dvalue

        return res


def save_obj(obj, outfile):
    # np.save(outfile, obj)
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(outfile):
    # return np.load(outfile)
    res = None
    with open(outfile, 'rb') as f:
        res = pickle.load(f)
    return res


def copy_array(arr):
    copy_arr = np.empty_like(arr)
    copy_arr[:] = arr
    return copy_arr


def read_image(path):
    bgr_image = cv2.imread(path)

    if len(bgr_image.shape) > 2 and bgr_image.shape[2] == 4:
        # remove 4th channel
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def resize_image(image, shape=(32, 32)):
    return cv2.resize(image, shape)


def plot_train_results(results, filepath=None):
    plt.figure(figsize=(8, 8))
    for result in results:
        label, (epochs, accuracy) = result
        plt.plot(epochs, accuracy, label=label)

    plt.title('Results')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def show_distribution(classes, n_classes, title, filepath=None):
    plt.figure(figsize=(6, 3))
    plt.title('Distribution for {}'.format(title))
    plt.hist(classes, bins=n_classes)
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    # TODO: uncomment it
    plt.tight_layout()
    plt.show()


def plot_features(features, classes=None, cols=None, title=None, filepath=None):
    features = features.squeeze()

    if cols is None:
        cols = len(features)

    rows = len(features) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.subplots_adjust(top=0.8)
    axs = axs.ravel()
    for row in range(rows):
        for col in range(cols):
            index = col + row * cols
            feature = features[index]
            axs[index].axis('off')
            if len(feature.shape) == 2:
                axs[index].imshow(feature, cmap="gray")
            else:
                axs[index].imshow(feature)

            if classes is not None:
                axs[index].set_title(classes[index])
    if title is not None:
        plt.suptitle(title, fontsize=12)

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    # TODO: uncomment it
    plt.tight_layout()
    plt.show()


def filter_by_class_id(features, classes, class_id):
    class_indexes = np.where(classes == class_id)[0]
    return features[class_indexes], classes[class_indexes]


def select_random_features(features, classes, size, class_id=None):
    in_features = features
    in_classes = classes

    if class_id is not None:
        in_features, in_classes = filter_by_class_id(features, classes, class_id)

    indices = np.random.choice(len(in_features), size=size, replace=False)
    return in_features[indices], in_classes[indices]


def select_first_n_features(features, classes, size, class_id=None):
    in_features = features
    in_classes = classes

    if class_id is not None:
        in_features, in_classes = filter_by_class_id(features, classes, class_id)

    indices = range(size)
    return in_features[indices], in_classes[indices]


def plot_first_n_features(features, labels, feature_class_id, features_size=10, title=None):
    if title is None:
        title = "{} - {}".format(feature_class_id, classes_catalog[feature_class_id])
    features_res, classes_res = select_first_n_features(features, labels, size=features_size,
                                                        class_id=feature_class_id)
    plot_features(features_res, classes_res, cols=features_size, title=title)


def plot_random_n_features(features, labels, rnd_features_size=10, rnd_classes_size=5):
    for idx in range(rnd_classes_size):
        feature_class_id = unique_classes[idx]
        title = "{} - {}".format(feature_class_id, classes_catalog[feature_class_id])
        features_res, classes_res = select_random_features(features, labels, size=rnd_features_size,
                                                           class_id=feature_class_id)
        plot_features(features_res, classes_res, cols=rnd_features_size, title=title)


def plot_two_feature_sets(orig_features, orig_classes, mod_features, mod_classes, cols=None, title=None, filepath=None):
    assert len(orig_features) == len(orig_classes) == len(mod_features) == len(mod_classes)
    assert orig_features.shape[1:] == mod_features.shape[1:]

    if cols is None:
        cols = len(orig_features)

    res_features = orig_features
    res_classes = orig_classes

    res_features = np.concatenate((res_features, mod_features), axis=0)
    res_classes = np.concatenate((res_classes, mod_classes), axis=0)

    plot_features(res_features, res_classes, cols=cols, title=title, filepath=filepath)


DATASET_URL = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
DATASET_NAMES_FILE = "signnames.csv"
OUTPUT_DIRECTORY = os.path.abspath("output")
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

input_files = get_input_files(DATASET_URL, OUTPUT_DIRECTORY)
classes_catalog = read_csv(DATASET_NAMES_FILE)

# Step 0: Load The Data
training_file = input_files["train"]
validation_file = input_files["valid"]
testing_file = input_files["test"]

assert os.path.exists(training_file)
assert os.path.exists(validation_file)
assert os.path.exists(testing_file)

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert len(X_train) == len(y_train)
assert len(X_valid) == len(y_valid)
assert len(X_test) == len(y_test)

# store input arrays for debugging purposes
# save_obj(X_train, os.path.join(OUTPUT_DIRECTORY, "X_train.npy"))
# save_obj(y_train, os.path.join(OUTPUT_DIRECTORY, "y_train.npy"))
# save_obj(X_valid, os.path.join(OUTPUT_DIRECTORY, "X_valid.npy"))
# save_obj(y_valid, os.path.join(OUTPUT_DIRECTORY, "y_valid.npy"))
# save_obj(X_test, os.path.join(OUTPUT_DIRECTORY, "X_test.npy"))
# save_obj(y_test, os.path.join(OUTPUT_DIRECTORY, "y_test.npy"))

# Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas
n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
unique_classes, unique_classes_size = np.unique(y_train, return_counts=True)
n_classes = len(unique_classes)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Include an exploratory visualization of the dataset
# TODO: uncomment me
# show_distribution(y_train, n_classes, 'Train Data')

datagen = ImageDataGenerator(
    fill_mode='nearest',
    horizontal_flip=False,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.15,
    rotation_range=10.0
)

classes_to_plot = 10
rnd_features, rnd_classes = select_random_features(X_train, y_train, classes_to_plot)
mod_rnd_features, mod_rnd_classes = datagen.flow(rnd_features, rnd_classes, batch_size=classes_to_plot,
                                                 shuffle=False).next()
mod_rnd_features = mod_rnd_features.astype(np.uint8)

# TODO: uncomment me
# plot_two_feature_sets(rnd_features, rnd_classes, mod_rnd_features, mod_rnd_classes)

train_images_enriched_npy = os.path.join(OUTPUT_DIRECTORY, "train_images_enriched.npy")
train_classes_enriched_npy = os.path.join(OUTPUT_DIRECTORY, "train_classes_enriched.npy")


def generated_samples_available():
    return os.path.exists(train_images_enriched_npy) and os.path.exists(train_classes_enriched_npy)


def load_generated_samples():
    return load_obj(train_images_enriched_npy), load_obj(train_classes_enriched_npy)


def save_generated_samples(images):
    (train_images_enriched, train_classes_enriched) = images
    save_obj(train_images_enriched, outfile=train_images_enriched_npy)
    save_obj(train_classes_enriched, outfile=train_classes_enriched_npy)


def validate_generated_data_consistency(in1, in2):
    (train_images_enriched1, train_classes_enriched1) = in1
    (train_images_enriched2, train_classes_enriched2) = in2

    assert np.array_equal(train_images_enriched1, train_images_enriched2)
    assert np.array_equal(train_classes_enriched1, train_classes_enriched2)


def generate_samples(features, classes, unique_classes, unique_classes_size, optimal_class_size=5000):
    res_features = copy_array(features)
    res_classes = copy_array(classes)

    # find avarage
    # optimal_size = np.mean(unique_classes_size)
    # find max number of occurrences across classes
    # optimal_class_size = unique_classes_size[np.argmax(unique_classes_size)]
    classes_with_missing_samples = np.where(unique_classes_size < optimal_class_size)[0]
    num_of_classes_with_missing_samples = len(classes_with_missing_samples)

    print("Classes to update: {}".format(num_of_classes_with_missing_samples))

    classes_processed = 0
    for idx, class_id in tqdm(enumerate(unique_classes)):
        num_of_occurrences = unique_classes_size[idx]
        if num_of_occurrences < optimal_class_size:
            missing_samples_size = int(optimal_class_size - num_of_occurrences)
            class_indexes = np.where(classes == class_id)[0]
            selected_features = features[class_indexes]
            selected_classes = classes[class_indexes]

            features_size = len(selected_features)
            quotient, remainder = divmod(missing_samples_size, features_size)

            if quotient > 0:
                for idx, new_images in enumerate(
                        datagen.flow(selected_features, selected_classes, batch_size=features_size, shuffle=False)):
                    (new_features, new_classes) = new_images
                    new_features = new_features.astype(np.uint8)

                    assert len(new_features) == len(new_classes) == features_size

                    res_features = np.concatenate((res_features, new_features), axis=0)
                    res_classes = np.concatenate((res_classes, new_classes), axis=0)

                    if idx + 1 == quotient:
                        break
            if remainder > 0:
                (new_features, new_classes) = datagen.flow(selected_features, selected_classes,
                                                           batch_size=remainder, shuffle=False).next()
                new_features = new_features.astype(np.uint8)

                assert len(new_features) == len(new_classes) == remainder

                res_features = np.concatenate((res_features, new_features), axis=0)
                res_classes = np.concatenate((res_classes, new_classes), axis=0)

            classes_processed += 1

    return res_features, res_classes


train_images_enriched = train_classes_enriched = None
if generated_samples_available():
    train_images_enriched, train_classes_enriched = load_generated_samples()
else:
    train_images_enriched, train_classes_enriched = generate_samples(X_train, y_train, unique_classes,
                                                                     unique_classes_size)
    save_generated_samples((train_images_enriched, train_classes_enriched))

print("Validate generated data consistency")
validate_generated_data_consistency((train_images_enriched, train_classes_enriched), load_generated_samples())
print("Done")

# Use enriched training dataset by default
X_train, y_train = train_images_enriched, train_classes_enriched

# TODO: uncomment it
# show_distribution(y_train, n_classes, 'Train Data')
# show_distribution(train_classes_enriched, n_classes, 'Generated Data')

# Pre-process the Data Set (normalization, grayscale, etc.)
train_images_normalized_npy = os.path.join(OUTPUT_DIRECTORY, "train_images_normalized.npy")
valid_images_normalized_npy = os.path.join(OUTPUT_DIRECTORY, "valid_images_normalized.npy")
test_images_normalized_npy = os.path.join(OUTPUT_DIRECTORY, "test_images_normalized.npy")


def normalized_images_available():
    return os.path.exists(train_images_normalized_npy) \
           and os.path.exists(valid_images_normalized_npy) \
           and os.path.exists(test_images_normalized_npy)


def load_normalized_images():
    return load_obj(train_images_normalized_npy), load_obj(valid_images_normalized_npy), \
           load_obj(test_images_normalized_npy)


def save_normalized_images(images):
    (train_images_normalized, valid_images_normalized, test_images_normalized) = images

    save_obj(train_images_normalized, train_images_normalized_npy)
    save_obj(valid_images_normalized, valid_images_normalized_npy)
    save_obj(test_images_normalized, test_images_normalized_npy)


def normalize_images_adapthist(images):
    # images = images[:10]
    images_gray = np.dot(images[:, :, :, :3], [0.2989, 0.587, 0.114])
    images_gray = images_gray / 255.0

    # beware, this modifies the input-argument
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in tqdm(range(len(images_gray))):
            images_gray[i] = exposure.equalize_adapthist(images_gray[i])

    images_gray = images_gray.reshape(-1, images_gray.shape[1], images_gray.shape[2], 1).astype(np.float32)
    return images_gray


def normalize_images_simple(images):
    images_gray_mean = np.mean(images, axis=3, keepdims=True)
    assert images_gray_mean.shape[:-1] == images.shape[:-1]

    images_gray_norm = (images_gray_mean - 128) / 128
    assert images_gray_norm.shape == images_gray_mean.shape

    return images_gray_norm


def normalize_images(images):
    # images = images[:10]
    return normalize_images_adapthist(images)


def validate_norm_data_consistency(in1, in2):
    (train_images_normalized1,
     valid_images_normalized1,
     test_images_normalized1) = in1
    (train_images_normalized2,
     valid_images_normalized2,
     test_images_normalized2) = in2

    assert np.array_equal(train_images_normalized1, train_images_normalized2)
    assert np.array_equal(valid_images_normalized1, valid_images_normalized2)
    assert np.array_equal(test_images_normalized1, test_images_normalized2)


train_images_normalized = valid_images_normalized = test_images_normalized = None

if normalized_images_available():
    print("Preloading data")
    train_images_normalized, valid_images_normalized, test_images_normalized = load_normalized_images()
else:
    print("Normalizing data")
    train_images_normalized = normalize_images(X_train)
    valid_images_normalized = normalize_images(X_valid)
    test_images_normalized = normalize_images(X_test)
    print("Saving normalized data")
    save_normalized_images((train_images_normalized, valid_images_normalized, test_images_normalized))

print("Validate normalized data consistency")
validate_norm_data_consistency((train_images_normalized, valid_images_normalized, test_images_normalized),
                               load_normalized_images())
print("Done")

# TODO: uncomment me
classes_to_plot = 3
# for idx in range(classes_to_plot):
#     feature_class_id = unique_classes[idx]
#     plot_first_n_features(X_train, y_train, feature_class_id,
#                           title="Original. {} - {}".format(feature_class_id, classes_catalog[feature_class_id]))
#     plot_first_n_features(train_images_normalized, y_train, feature_class_id,
#                           title="Grayscale normalized. {} - {}".format(feature_class_id,
#                                                                        classes_catalog[feature_class_id]))

# Model Architecture¶
EPOCHS = 20
BATCH_SIZE = 128
(height, width, depth) = train_images_normalized.shape[1:]


def lenetMultiDropoutsBefore45(x, keep_prob, n_classes):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    dr0 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(dr0, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    dr1 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = {n_classes}.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    # logits = tf.matmul(fc2, fc3_W) + fc3_b
    logits = tf.matmul(dr1, fc3_W) + fc3_b

    return logits, {"conv1": conv1, "conv2": conv2, "fc0": fc0, "fc1": fc1, "fc2": fc2, "logits": logits}


def evaluate(X_data, y_data, accuracy_operation, x, y, keep_prob, keep_prob_value=1.0):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob_value})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# Train, Validate and Test the Model
def train_model(train_features, train_classes, valid_features, valid_classes, model_config, keep_prob_value=0.5,
                model_name="lenet", desc=None, output_dir=OUTPUT_DIRECTORY):
    print("Train model {}".format(model_name))

    if desc is None:
        desc = model_name

    x, y, keep_prob, logits, training_operation, accuracy_operation, cnn_params = model_config()
    saver = tf.train.Saver()

    results = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")
        print()
        for epoch in range(EPOCHS):
            train_features, train_classes = shuffle(train_features, train_classes)
            for offset in tqdm(range(0, len(train_features), BATCH_SIZE)):
                end = offset + BATCH_SIZE
                batch_x, batch_y = train_features[offset:end], train_classes[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob_value})

            validation_accuracy = evaluate(valid_features, valid_classes, accuracy_operation, x, y, keep_prob)
            epoch_number = epoch + 1
            print("EPOCH {} ...".format(epoch_number))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            results.append([epoch_number, validation_accuracy])

        saver.save(sess, os.path.join(output_dir, model_name))
        print("Model saved")

        # trasform [[epoch, accuracy], [epoch, accuracy]] -> [[epoch, epoch], [accuracy, accuracy]]
        return desc, np.array(results).T


def calc_top_k(new_features, new_classes, model_config, top_k_value=5,
               keep_prob_value=1.0, model_name="lenet", desc=None, output_dir=OUTPUT_DIRECTORY):
    print("Calculate top_k model {}".format(model_name))

    if desc is None:
        desc = model_name

    model_file = os.path.join(output_dir, model_name)

    x, y, keep_prob, logits, training_operation, accuracy_operation, cnn_params = model_config()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)
        sess = tf.get_default_session()
        top_k_results = sess.run(tf.nn.top_k(tf.nn.softmax(logits), top_k_value),
                                 feed_dict={x: new_features, keep_prob: keep_prob_value})
        return top_k_results


def calc_probability(new_features, new_classes, model_config,
                     keep_prob_value=1.0, model_name="lenet", desc=None, output_dir=OUTPUT_DIRECTORY):
    print("Calculate probability model {}".format(model_name))

    if desc is None:
        desc = model_name

    model_file = os.path.join(output_dir, model_name)

    x, y, keep_prob, logits, training_operation, accuracy_operation, cnn_params = model_config()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)
        sess = tf.get_default_session()
        predicted_classes = sess.run(tf.argmax(logits, 1), feed_dict={x: new_features, keep_prob: keep_prob_value})
        assert new_classes.shape == predicted_classes.shape
        return predicted_classes


def model_multi_dropouts_before45_config():
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, (None, height, width, depth), name="X")
    y = tf.placeholder(tf.int32, (None), name="Y")
    one_hot_y = tf.one_hot(y, n_classes)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    rate = 0.001
    logits, cnn_params = lenetMultiDropoutsBefore45(x, keep_prob, n_classes)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, keep_prob, logits, training_operation, accuracy_operation, cnn_params


model_metrics_file = os.path.join(OUTPUT_DIRECTORY, "model_metrics.pickle")


def model_metrics_available():
    return os.path.exists(model_metrics_file)


def save_model_metrics(metrics):
    save_obj(metrics, model_metrics_file)


def load_model_metrics():
    return load_obj(model_metrics_file)


def plot_confusion_matrix(cm):
    cm = [row / sum(row) for row in cm]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.show()


def print_confusion_matrix(cm, orig_classes, catalog=classes_catalog, order_desc=False):
    results = []
    for i, row in enumerate(cm):
        accuracy = round(row[i] / sum(row) * 100, 2)
        class_id = orig_classes[i]
        class_name = catalog[orig_classes[i]]
        results.append((accuracy, class_id, class_name))

    # results = [(row[i] / sum(row) * 100, orig_classes[i], catalog[orig_classes[i]]) for i, row in enumerate(cm)]
    total_accuracy = []
    for result in sorted(results, key=lambda x: x[0], reverse=order_desc):
        (accuracy, class_id, class_name) = result
        print('{}% - {} / {}'.format(accuracy, class_id, class_name))
        total_accuracy.append(accuracy)

    print("Accuracy: Mean: {}% Std: {}".format(np.around(np.mean(total_accuracy), decimals=3),
                                               np.around(np.std(total_accuracy), decimals=3)))


train_setup = [
    partial(train_model, train_features=train_images_normalized, train_classes=y_train,
            valid_features=valid_images_normalized,
            valid_classes=y_valid, model_config=model_multi_dropouts_before45_config,
            desc="NDR 45 Grayscale normalized", model_name="ndr45_train_images_normalized_model"),
]

# TODO: uncomment me
# results = []
# for test in train_setup:
#     result = test()
#     print("Result: ", result)
#     results.append(result)
#
# print("Saving model metrics")
# save_model_metrics(results)
# print("Done")

# results = load_model_metrics()
# plot_train_results(results, os.path.join(OUTPUT_DIRECTORY, "train_results.png"))

# run model on test data
test_features_predicted_classes = calc_probability(test_images_normalized, y_test,
                                                   model_config=model_multi_dropouts_before45_config,
                                                   desc="NDR 45 Grayscale normalized (Test)",
                                                   model_name="ndr45_train_images_normalized_model")

print_confusion_matrix(confusion_matrix(y_test, test_features_predicted_classes), y_test, order_desc=True)


# plot_confusion_matrix(confusion_matrix(y_test, test_features_predicted_classes))


def plot_top_k(in_features, in_classes, top_k):
    rows = top_k[0].shape[0]
    cols = top_k[0].shape[1] + 1

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    fig.subplots_adjust(top=0.8)
    axs = axs.ravel()
    for row in range(rows):
        for col in range(cols):
            index = col + row * cols
            feature = None
            class_id = -1
            predicted_prcnt = -1
            if col == 0:
                feature = in_features[row].squeeze()
                class_id = in_classes[row]
            else:
                class_id = top_k[1][row][col - 1]
                predicted_prcnt = 100 * top_k[0][row][col - 1]
                class_indexes = np.where(y_test == class_id)[0]
                feature = test_images_normalized[class_indexes][0].squeeze()
            axs[index].axis('off')
            if len(feature.shape) == 2:
                axs[index].imshow(feature, cmap="gray")
            else:
                axs[index].imshow(feature)

            if col == 0:
                axs[index].set_title('Original: {}'.format(class_id))
            else:
                axs[index].set_title('Predicted: {} ({:.0f}%)'.format(class_id, predicted_prcnt))
    plt.show()


# run model on random data
# print()
# print("Available classes")
# for idx, class_id in enumerate(unique_classes):
#     print("{} - {}".format(class_id, classes_catalog[idx]))

new_features_with_classes = [(resize_image(read_image(path)), int(basename(path).split(".")[-2])) for path in
                             glob.glob('images/*.jpg')]
(new_features, new_classes) = zip(*new_features_with_classes)
new_features = np.array(new_features).astype(np.float32)
new_classes = np.array(new_classes)

assert new_features.shape[1:] == X_train.shape[1:]

original_features = []
original_classes = []
for idx, class_id in enumerate(new_classes):
    class_indexes = np.where(y_test == class_id)[0]
    original_features.append(X_test[class_indexes][0].squeeze())
    original_classes.append(class_id)

original_features = np.array(original_features)
original_classes = np.array(original_classes)

# plot_two_feature_sets(original_features.astype(np.uint8), original_classes, new_features.astype(np.uint8), new_classes,
#                       title="Original/Random")

new_features_normalized = normalize_images(new_features)
# plot_features(new_images, new_labels, title="Original Random Images")
# plot_features(new_images_normalized, new_labels, title="Normalized Random Images")

print("Test new images")
new_features_predicted_classes = calc_probability(new_features_normalized,
                                                  new_classes,
                                                  model_config=model_multi_dropouts_before45_config,
                                                  desc="NDR 45 Grayscale normalized (New Images)",
                                                  model_name="ndr45_train_images_normalized_model")
print_confusion_matrix(confusion_matrix(new_classes, new_features_predicted_classes), new_classes, order_desc=True)

top_k_results = calc_top_k(new_features_normalized, new_classes,
                           model_config=model_multi_dropouts_before45_config,
                           desc="NDR 45 Grayscale normalized (New Images)",
                           model_name="ndr45_train_images_normalized_model")


# plot_top_k(new_features_normalized, new_classes, top_k_results)

def outputFeatureMap(sess, x, keep_prob, keep_prob_value, image_input, tf_activation, activation_min=-1,
                     activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess, feed_dict={x: image_input, keep_prob: keep_prob_value})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 15))
    for featuremap in range(featuremaps):
        plt.subplot(6, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                       vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")
    plt.show()


def plot_feature_map(in_features, in_classes, model_config,
                     keep_prob_value=1.0, load_model=True, conv_name="conv1",
                     model_name="lenet", desc=None, output_dir=OUTPUT_DIRECTORY):
    print("Calculate probability model {}".format(model_name))

    if desc is None:
        desc = model_name

    model_file = os.path.join(output_dir, model_name)

    x, y, keep_prob, logits, training_operation, accuracy_operation, cnn_params = model_config()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if load_model:
            saver.restore(sess, model_file)
        sess = tf.get_default_session()
        outputFeatureMap(sess, x, keep_prob, keep_prob_value, in_features, cnn_params[conv_name])


plot_feature_map(new_features_normalized, new_classes, model_config=model_multi_dropouts_before45_config,
                 desc="NDR 45 Grayscale normalized (New Images)",
                 model_name="ndr45_train_images_normalized_model", load_model=False, conv_name="conv1")

plot_feature_map(new_features_normalized, new_classes, model_config=model_multi_dropouts_before45_config,
                 desc="NDR 45 Grayscale normalized (New Images)",
                 model_name="ndr45_train_images_normalized_model", conv_name="conv1")

plot_feature_map(new_features_normalized, new_classes, model_config=model_multi_dropouts_before45_config,
                 desc="NDR 45 Grayscale normalized (New Images)",
                 model_name="ndr45_train_images_normalized_model", load_model=False, conv_name="conv2")

plot_feature_map(new_features_normalized, new_classes, model_config=model_multi_dropouts_before45_config,
                 desc="NDR 45 Grayscale normalized (New Images)",
                 model_name="ndr45_train_images_normalized_model", conv_name="conv2")
