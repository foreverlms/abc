#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
import os.path

import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple
import cpp_demangle


def plot_object(object_size_file_path, save_dir):
    kbs_list = []
    file_names_list = []
    with open(object_size_file_path, 'r') as f:
        for line in f.readlines():
            kbs, file_name = line.split()
            # kilobytes
            kbs = float(kbs[:-2])
            file_name = file_name[9:-1]
            kbs_list.append(kbs)
            file_names_list.append(file_name)

    data_dict = dict(zip(file_names_list, kbs_list))

    # none zero data
    none_zero_data = [data_dict[key] for key in file_names_list if data_dict[key] > 0]
    none_zero_names = [key for key in file_names_list if data_dict[key] > 0]
    data = {
        "Kilobytes": pd.Series(none_zero_data),
        "ObjectFile": pd.Series(none_zero_names)
    }
    df = pd.DataFrame(data)

    df = df.set_index("ObjectFile")

    axes = df.plot(kind='barh', figsize=(20, 15))
    axes.bar_label(axes.containers[0])  # bar显示对应值
    fig = axes.get_figure()
    fig.savefig("/Users/bob/Desktop/object_files.jpeg", dpi=300)

    # zero data
    zero_data_name = [key for key in data_dict.keys() if not data_dict[key] > 0]
    print(zero_data_name)

    # all datas
    data = {
        "Kilobytes": pd.Series(kbs_list),
        "ObjectFile": pd.Series(file_names_list)
    }
    df = pd.DataFrame(data)
    df = df.set_index("ObjectFile")
    df.to_csv(os.path.join(save_dir, "object_file.csv"))


def demangle(symbol):
    return cpp_demangle.demangle(symbol)


def process_top_200(file_path, save_dir):
    symbol_sizes = []
    symbol_name = []
    with open(file_path, 'r') as f:
        content = f.read()
        data = make_tuple(content)
        for element in data:
            size, symbol = element.split()
            size = size[:-2]
            symbol = demangle(symbol)
            symbol_sizes.append(size)
            symbol_name.append(symbol)
    data = {
        "Kilobytes": pd.Series(symbol_sizes),
        "SymbolName": pd.Series(symbol_name)
    }
    df = pd.DataFrame(data)
    df = df.set_index("SymbolName")
    df.to_csv(os.path.join(save_dir, "symbols_size.csv"))


if __name__ == '__main__':
    # 出图，每个.o文件大小
    pnn_size_dir = "/Users/bob/Downloads/PNN_size/"
    plot_object("/Users/bob/Downloads/PNN_size/object_file.txt", pnn_size_dir)
    process_top_200("/Users/bob/Downloads/PNN_size/top_200.txt", pnn_size_dir)
