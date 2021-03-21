import numpy as np
from math import floor, sqrt
from collections import Counter
from os import listdir, getcwd
from os.path import isfile, join

NO_OF_FOLDS = 5
NO_OF_NEIGHBORS = 11


def read_dataset(path):
    bin_imgs = [f for f in listdir(path) if isfile(join(path, f))]
    input_vec = np.zeros((len(bin_imgs), 1024), dtype=int)
    output_label = np.zeros(len(bin_imgs), dtype=int)
    for index, bin_imgs_index in enumerate(bin_imgs):
        file_string = open(join(path, bin_imgs_index),
                           "r").read().replace("\n", '')
        arr_img = np.array(list(file_string))
        current_label = np.array(bin_imgs_index.split("_")[1])
        input_vec[index, :] = arr_img
        output_label[index] = current_label
    return input_vec, output_label


def dataset_shuffler(input_vec, output_label):
    rows, cols = input_vec.shape
    range_array = np.arange(rows)
    perm_arr = np.random.permutation(range_array)

    shuff_input_vec = input_vec[perm_arr, :]
    shuff_output_label = output_label[perm_arr]

    return shuff_input_vec, shuff_output_label


def train_validate(input_vec, output_label):
    print(input_vec.shape)
    print(output_label.shape)

    neighbour_iters = int((NO_OF_NEIGHBORS + 1) / 2)

    rows_ip_vec, cols_ip_vec = input_vec.shape
    safe_fold_size = floor(rows_ip_vec / NO_OF_FOLDS)
    k_and_error = np.empty(NO_OF_FOLDS, dtype=object)

    fold_iter = 0
    for fold in range(NO_OF_FOLDS):
        print("=" * 50)
        print(f"Current fold: {fold + 1}")
        print("_" * 40)
        train_ip = np.zeros(
            (rows_ip_vec - safe_fold_size, cols_ip_vec), dtype=int)
        train_ol = np.zeros(rows_ip_vec - safe_fold_size, dtype=int)

        validate_ip = np.zeros((safe_fold_size, cols_ip_vec), dtype=int)
        validate_ol = np.zeros(safe_fold_size, dtype=int)

        if fold == 0:
            train_ip = input_vec[safe_fold_size:, :]
            train_ol = output_label[safe_fold_size:]

            validate_ip = input_vec[:safe_fold_size, :]
            validate_ol = output_label[:safe_fold_size]
        elif fold == 4:
            train_ip = input_vec[: 4 * safe_fold_size, :]
            train_ol = output_label[: 4 * safe_fold_size]

            validate_ip = input_vec[4 *
                                    safe_fold_size:, :]
            validate_ol = output_label[4 *
                                       safe_fold_size:]
        else:
            train_ip = np.vstack(
                (input_vec[: fold * safe_fold_size, :], input_vec[(fold + 2) * safe_fold_size:, :]))
            print(train_ip.shape)
            train_ol = np.concatenate(
                (output_label[: fold * safe_fold_size], output_label[(fold + 2) * safe_fold_size:]))

            validate_ip = input_vec[fold *
                                    safe_fold_size: (fold + 1) * safe_fold_size, :]
            validate_ol = output_label[fold *
                                       safe_fold_size: (fold + 1) * safe_fold_size]

        print(f"Training for fold: {fold + 1}")

        fold_k_error = np.empty(neighbour_iters - 1, dtype=object)
        iteration = 0
        for k in range(3, NO_OF_NEIGHBORS + 1):
            if k % 2 != 0:
                print(f"k is {k}")
                predictions_train = np.zeros(train_ol.shape[0])
                for index, train_ip_index in enumerate(train_ip):
                    distances = np.array([euclidean_dist(train_ip_index, current_train_ip)
                                          for current_train_ip in train_ip])
                    k_nearest_indices = np.argsort(distances)[:k]
                    k_nearest_labels = [train_ol[i]
                                        for i in k_nearest_indices]

                    most_common = Counter(k_nearest_labels).most_common(1)
                    predictions_train[index] = most_common[0][0]

                error = np.sum(predictions_train != train_ol)
                fold_k_error[iteration] = (k, error)
                train_acc = np.sum(predictions_train ==
                                   train_ol) / len(train_ol) * 100
                print(
                    f"Classification errors: {error}, Training set accuracy: {train_acc}")
                iteration += 1

        print("_" * 40)
        print(f"Validating for fold: {fold + 1}")

        best_fold_k = min(fold_k_error, key=lambda t: t[1])
        predictions_validate = np.zeros(validate_ol.shape[0])
        for index, validate_ip_index in enumerate(validate_ip):
            distances = np.array([euclidean_dist(validate_ip_index, current_validate_ip)
                                  for current_validate_ip in validate_ip])
            k_nearest_indices = np.argsort(distances)[:best_fold_k[0]]
            k_nearest_labels = [validate_ol[i]
                                for i in k_nearest_indices]

            most_common = Counter(k_nearest_labels).most_common(1)
            predictions_validate[index] = most_common[0][0]

        error = np.sum(predictions_validate !=
                       validate_ol)
        valid_acc = np.sum(predictions_validate ==
                           validate_ol) / len(validate_ol)
        print(
            f"Classification errors: {error} Validation set accuracy: {valid_acc}")
        k_and_error[fold_iter] = (best_fold_k[0], error)
        fold_iter += 1

    best_k = min(k_and_error, key=lambda t: t[1])
    return best_k


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))


def test(test_ip, test_ol, train_ip, train_ol, k):
    rows_ip_vec, cols_ip_vec = test_ip.shape
    predictions_test = np.zeros(test_ol.shape[0])
    print("=" * 50)
    print(f"Testing for given test dataset")
    print("_" * 40)
    for index, test_ip_index in enumerate(test_ip):
        distances = np.array([euclidean_dist(test_ip_index, current_test_ip)
                              for current_test_ip in test_ip])

        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = [test_ol[i]
                            for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        predictions_test[index] = most_common[0][0]
    errors = np.sum(predictions_test != test_ol)
    print(f"Classification errors for testing dataset: {errors}")
    train_acc = np.sum(predictions_test == test_ol) / len(test_ol) * 100

    return train_acc


def main():
    train_ip_vec, train_op_vec = read_dataset(
        join(getcwd(), "binary-hand-digits", "training_validation"))
    test_ip_vec, test_op_vec = read_dataset(
        join(getcwd(), "binary-hand-digits", "test"))
    shuff_train_ip_vec, shuff_train_op_vec = dataset_shuffler(
        train_ip_vec, train_op_vec)
    shuff_test_ip_vec, shuff_test_op_vec = dataset_shuffler(
        test_ip_vec, test_op_vec)
    best_k = train_validate(shuff_train_ip_vec, shuff_train_op_vec)
    testing_accuracy = test(shuff_test_ip_vec, shuff_test_op_vec,
                            shuff_train_ip_vec, shuff_train_op_vec, best_k[0])

    print(
        f"Testing accuracy with the the best yet k, k=3 is {testing_accuracy}%")


if __name__ == "__main__":
    main()
