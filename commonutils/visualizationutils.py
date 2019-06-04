from six import string_types
import seaborn as sns
from matplotlib import pyplot as plt


def plot_bar_chart(x,y,title='Sample Title',dataframe=None,filename_to_save='sample.png'):
    plt.figure()
    sns.set(style="whitegrid")
    if type(x) is list and type(y) is list and len(x) == len(y):
        ax = sns.barplot(x,y)
        ax.set_title(title)
        plt.savefig(filename_to_save)
        plt.close()
    elif dataframe is not None and isinstance(x,string_types) and isinstance(y,string_types):
        ax = sns.barplot(x=x,y=y,data=dataframe)
        ax.set_title(title)
        plt.savefig(filename_to_save)
        plt.close()
    else:
        raise Exception("Configuration ill specified dude !!!")


def plot_histogram_chart(data_iterable,data_label='Sample Label',n_bins=3,
                         title_name='Sample Histogram', filename_to_save='sample_hist.png',
                         x_label='Sample X Label',y_label='Sample Y Label'):
    plt.figure()
    ax = sns.distplot(data_iterable, kde=False, rug=False, hist=True, norm_hist=False, bins=n_bins,
                 label=data_label)
    ax.set_title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename_to_save)
    plt.close()


def plot_boxplot_chart(x_name,y_name,dataframe,filename_to_save='sample_box_plot.png'):
    plt.figure()
    sns.boxplot(x=x_name, y=y_name,data=dataframe)
    plt.savefig(filename_to_save)
    plt.close()


def plot_violinstrip_chart(x_name, y_name,dataframe, filename_to_save='sample_violin_strip.png'):
    plt.figure()
    sns.violinplot(x=x_name, y=y_name, data=dataframe, inner=None, color="0.8")
    sns.stripplot(x=x_name, y=y_name,data=dataframe, jitter=True)
    plt.savefig(filename_to_save)
    plt.close()


def plot_heatmap_chart(pivoted_dataframe,filename_to_save='sample_heatmap.png'):
    plt.figure()
    sns.heatmap(pivoted_dataframe)
    plt.savefig(filename_to_save)
    plt.close()


def plot_stacked_barchart(x_name,y_name,dataframe,groupby,filename_to_save='sample_stacked_barplot.png'):
    plt.figure()
    sns.barplot(x=x_name, y=y_name, hue=groupby, data=dataframe, capsize=0.0, errwidth=0.0)
    plt.savefig(filename_to_save)
    plt.close()

