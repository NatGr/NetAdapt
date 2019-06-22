"""computes the table associated with a wideresnet Network, this script is quite complex bacause we tested many
different methods to compute the tables"""

import numpy as np
import os
import gc
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # forces tf to run on cpu, which is what we want to do here
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.layers import Input
from .measure_fcts import save_tflite_file, get_measure_tf_lite_file
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, Activation, Flatten


def make_conv_model(inputs, out_channels, stride, kernel_size=3):
    """creates a small sequential model composed of a convolution, a batchnorm and a relu activation"""
    outputs = Conv2D(out_channels, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    return Model(inputs=inputs, outputs=outputs)


def make_fc_model(inputs, num_classes, width):
    """creates a small sequential model composed of an average pooling and a fully connected layer"""
    outputs = AveragePooling2D(pool_size=width)(inputs)
    outputs = Flatten()(outputs)
    outputs = Dense(units=num_classes)(outputs)
    return Model(inputs=inputs, outputs=outputs)


def compute_perf_table_wrn_2_times(args):
    if keras_backend.image_data_format() != 'channels_last':
        raise ValueError('channels_last data format expected')  # channels_last is said to run faster on cpu

    tmp_folder = "{}_{}".format(args.tmp_folder, args.offset_process)

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    tmp_keras_file = os.path.join(tmp_folder, 'model{}.h5'.format(args.offset_process))

    perf_table = {}
    if args.img_size == 32:
        strides = [1, 1, 2, 2]
        cum_strides = np.cumprod(strides)  # cumulated strides from start
    else:
        raise ValueError('unsupported input resolution')
    # same as in wideresnet.py, needs to be copied not to have to use an env. with both pytorch and tf
    n_channels = [16, int(16 * args.width), int(32 * args.width), int(64 * args.width)]
    fm_sizes = [int(args.img_size // cum_strides[i]) for i in range(cum_strides.shape[0])]

    compute_table_on = [("Conv_0", fm_sizes[0], 3, n_channels[0], strides[0]),
                        ("FC", fm_sizes[3], n_channels[3], 1, None)]
    for i in range(1, 4):
        compute_table_on.append(("Stride_" + str(i), fm_sizes[i - 1], n_channels[i - 1], n_channels[i], strides[i]))
        # used for Conv_i_0_1
        compute_table_on.append(("No_Stride_" + str(i), fm_sizes[i], n_channels[i], n_channels[i], 1))
        # used for Conv_i_j_1 and Conv_i_0_2

    for i in range(1, 4):  # Skip_i
        compute_table_on.append(("Skip_" + str(i), fm_sizes[i - 1], n_channels[i - 1], n_channels[i], strides[i]))

    for i, (name, width, max_in_channels, max_out_channels, stride) in enumerate(compute_table_on):
        table_entry = np.zeros((max_in_channels, max_out_channels))
        print("{} tables out of {} done".format(i, len(compute_table_on)))
        print(name, width, max_in_channels, max_out_channels, stride)

        for in_channels in range(1, max_in_channels + 1):
            print("{} input_channels out of {}".format(in_channels, max_in_channels))

            # determines the fraction of the work we will do in this process if there is several of them
            if args.num_process > 1 and args.mode == 'save':
                step = max_out_channels / args.num_process
                out_channels_range = range(round(step * args.offset_process) + 1,
                                           round(step * (args.offset_process + 1)) + 1)
            else:
                out_channels_range = range(1, max_out_channels + 1)

            for out_channels in out_channels_range:
                tflite_file = os.path.join(args.output_folder, "{}_{}_{}.tflite".format(i, in_channels, out_channels))

                if args.mode == "save":
                    inputs = Input(shape=(width, width, in_channels))
                    if name == "FC":
                        model = make_fc_model(inputs, args.num_classes, width)
                    else:
                        model = make_conv_model(inputs, out_channels, stride, kernel_size=1 if name[:4] == "Skip"
                                                else 3)

                    save_tflite_file(model, tmp_keras_file, tflite_file)
                    del model
                    keras_backend.clear_session()
                    gc.collect()

                elif args.mode == "load":
                    table_entry[in_channels - 1, out_channels - 1] = get_measure_tf_lite_file(
                        tflite_file, number_of_measures=args.num_measures, benchmark_loc=args.benchmark_lite_loc)

        perf_table[name] = table_entry
    return perf_table  # returns a useless table when args.mode == "save"
