"""file containing the functions used to measure the performance of a model using tensorflow"""


import subprocess
from tensorflow import lite
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import SGD


def save_tflite_file(model, tmp_keras_file, tflite_file):
    """given a model, saves that model as a .tflite file (necessists to first save it a sa keras file)"""
    model.compile(optimizer=SGD(), loss='binary_crossentropy')
    save_model(model, tmp_keras_file)

    # Convert to TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_keras_model_file(tmp_keras_file)
    tflite_model = converter.convert()
    with open(tflite_file, "wb") as file:
        file.write(tflite_model)


def get_measure_tf_lite_file(tflite_file, number_of_measures, benchmark_loc):
    """given a tflite file, benchmarks the time needed for a prediction in C++ using the benchmark tool associated with
    tf-lite (this tool does not return median but only mean so we will use that instead)
    :return: the mean of number_of_measures trials"""

    command_line = "{} --graph={} --min_secs=0 --warmup_min_secs=0 --num_runs={} |& tr -d '\n' | awk {}".format(
        benchmark_loc, tflite_file, number_of_measures, "'{print $NF}'")  # tr removes the \n and awk gets
    # the last element of the outputs message, |& is used before tr because we want to pipe strderr and not
    # stdout
    result = float(subprocess.check_output(command_line, shell=True, executable='/bin/bash')) / 10 ** 6  # result
    # given in microseconds
    return result
