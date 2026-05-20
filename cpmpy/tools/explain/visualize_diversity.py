
from matplotlib import pyplot as plt
import matplotlib_venn
from upsetplot import UpSet, from_contents
import seaborn as sns


def visualize_venn(sets, label="MUS"):
    """
        Visualize the overlap of a collection of sets in a Venn diagram.

        :param: sets: the list or set of sets to visualize
        :param: label: the label for the sets (e.g. "MUS", "MCS", "MSS", ...)
    """
    if len(sets) == 2:
        matplotlib_venn.venn2(sets, set_labels=(f"{label} 1", f"{label} 2"))
    elif len(sets) == 3:
        matplotlib_venn.venn3(sets, set_labels=(f"{label} 1", f"{label} 2", f"{label} 3"))
    else:
        raise ValueError(f"Venn diagrams only support 2 or 3 sets. Received {len(sets)} sets.")
    plt.show()


def visualize_upset(sets, label="MUS"):
    """
        Create an UpSet plot of a collection of sets. UpSet plots visualize the intersections between sets.

        :param: sets: the list or set of sets to visualize
        :param: label: the label for the sets (e.g. "MUS", "MCS", "MSS", ...)
    """
    labels = [f"{label} {i+1}" for i in range(len(sets))]
    upset_data = from_contents(dict(zip(labels,sets)))
    UpSet(upset_data).plot()
    plt.show()


def visualize_heatmap(diversity_matrix, colormap="YlOrRd", title=None, save_path=None):
    """
        Visualize the diversity between all the pairs of sets by creating a heatmap of the diversity matrix.

        :param: diversity_matrix: a diversity matrix (upper triangular)
        :param: colormap: a seaborn colormap
        :param: save_path: optional file path to save the figure (e.g. "fig.pdf")
    """
    with sns.axes_style("white"):
        ax = sns.heatmap(diversity_matrix, vmax=1, square=True, cmap=colormap)
        ax.collections[0].colorbar.set_label("diversity")
        plt.xlabel("MUS number (order of generation)")
        plt.ylabel("MUS number (order of generation)")
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()