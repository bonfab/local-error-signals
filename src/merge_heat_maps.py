import os

import pandas as pd

from evaluate_dimensions import plot_rdms, plot_cka


def get_common_name(file_name):
    return file_name.split(".")[0].split("-")[0]


def set_rownames_as_index(df):
    new_df = df.reindex(df.columns)
    new_df[new_df.columns] = df.values
    return new_df




def parse_and_merge(path_dir):
    csvs = os.listdir(path_dir)
    merged = []
    for csv1 in csvs:
        if csv1 in merged:
            continue

        merged.append(csv1)
        names = csv1.split("_")
        name1 = names[0]
        name2 = names[1]
        df = pd.read_csv(os.path.join(path_dir, csv1))
        count = 1
        for csv2 in csvs:
            if get_common_name(csv1) == get_common_name(csv2):
                merged.append(csv2)
                df = df.add(pd.read_csv(os.path.join(path_dir, csv2)))
                count += 1
        df = df / count
        yield df, name1, name2, path_dir, get_common_name(csv1) + "-merged.png"


if __name__ == "__main__":
    path_dir = "../13-34-54/repr_analysis_results/rdms/csvs"
    """for df, name1, name2, path_d, file_name in parse_and_merge(path_dir):
        df = set_rownames_as_index(df)
        plot_rdms(df, name1, name2, path_d, file_name)"""

    path_dir = "../13-34-54/repr_analysis_results/cka/csvs"
    for df, name1, name2, path_d, file_name in parse_and_merge(path_dir):
        df = set_rownames_as_index(df)
        plot_cka(df, name1, name2, path_d, file_name)
