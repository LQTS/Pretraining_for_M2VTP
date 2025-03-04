
## 要求：
# 1. 输入多个训练的保存的log文件地址，绘制一个key在一张表中
# 2. 可随意改名

import os.path

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
from logger import DataLog
from matplotlib.pyplot import MultipleLocator
import argparse
# import bar_chart_race as bcr
import pandas as pd

from pathlib import Path
from typing import List

def make_plots(
    log_files_path_list: List[str],
    labels: List[str],
    y_key: str,
    save_path: str,
    title: str = 'performance of different methods',
    y_label: str = None
) -> None:
    assert log_files_path_list is not None
    assert y_key is not None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    if not isinstance(log_files_path_list, list):
        log_files_path_list = list(log_files_path_list)

    for idx, log_file in enumerate(log_files_path_list):
        # set params
        key_data = []
        # read data
        logger = DataLog()
        logger.read_log(log_file)
        log = logger.log
        if y_key in log.keys():
            key_data.append(log[y_key])
        else:
            raise KeyError(f"No key \"{y_key}\" in log files \"{log_file}\"!")

        min_len = min(min([np.array(data).shape[-1] for data in key_data]), 800)
        curve_data = np.array([np.array(data[:min_len]) for data in key_data])

        mean = np.mean(curve_data, axis=0)
        std = curve_data.std(axis=0) / np.sqrt(curve_data.shape[0])

        x_axis = np.arange(min_len)
        # ax.plot(x_axis, mean, color=colors[j], label=labels[j])
        # plt.fill_between(x_axis, mean + std, mean - std, color=colors[j], alpha=0.2)
        ax.plot(x_axis, mean, label=labels[idx])
        plt.fill_between(x_axis, mean + std, mean - std, alpha=0.2)

    plt.title(title, fontsize=20)
    plt.grid(True, which='both')
    plt.ylabel(y_key if y_label is None else y_label, fontsize=20)
    plt.xlabel('Iteration', fontsize=20)
    plt.xticks(size=14)
    plt.yticks(size=14)

    plt.legend()
    plt.savefig(save_path, dpi=500)
    plt.show()

def main():
    root_dir = "D:\\MyProgram\\ViTacMani\\pretrain\\vitac_pretrain\\"
    train_proj = {
        # "VT2T-ReAll": root_dir + "vt2t-reall-bin-ft+dataset-BottleCap.csv",
        # "VT2T-ReAll-NoTac": root_dir + "vt2t-reall-bin-notac-ft+dataset-BottleCap.csv",
        # "VT2T-RePic": root_dir + "vt2t-repic-bin-ft+dataset-BottleCap.csv",
        # "VT2T-RePic-NoTac": root_dir + "vt2t-repic-bin-notac-ft+dataset-BottleCap.csv",
        "VT5T-ReAll": root_dir + "vt5t-reall-bin-ft+dataset-BottleCap.csv",
        "VT5T-ReAll-NoTac": root_dir + "vt5t-reall-bin-notac-ft+dataset-BottleCap.csv",
        # "VT5T-RePic": root_dir + "vt5t-repic-bin-ft+dataset-BottleCap.csv",
        # "VT5T-RePic-NoTac": root_dir + "vt5t-repic-bin-notac-ft+dataset-BottleCap.csv",

    }
    Title = "val_tac_loss"

    log_files_path_list = list(train_proj.values())
    labels = list(train_proj.keys())
    make_plots(
        log_files_path_list,
        labels,
        y_key=Title,
        save_path=root_dir + str(Title),
        title=Title
    )

if __name__ == "__main__":
    main()