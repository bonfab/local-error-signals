import os

import seaborn as sns
import pandas as pd


def make_acc_plots(path_to_csv="./training_results.csv"):
    results = pd.read_csv(path_to_csv)
    plot = sns.lineplot(data=pd.melt(results[['epoch', 'train_acc', 'valid_acc']], 'epoch'),
                        x='epoch',
                        y='value',
                        hue='variable')
    fig = plot.get_figure()
    fig.savefig(
        os.path.join(os.path.split(path_to_csv)[0], "acc_plot.png")
    )
