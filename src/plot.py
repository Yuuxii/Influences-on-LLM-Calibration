import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

old_model_names = ['gemma2', 'phi4', 'llama7', 'phi7', 'llama8', 'llama31-8',  'gemma9',  
                   'google_gemma-2-27b-it', 'mistralai_Mixtral-8x7B-Instruct-v0.1', 'meta-llama_Meta-Llama-3-70B-Instruct', 'meta-llama_Meta-Llama-3.1-70B-Instruct', 'Qwen_Qwen2-72B-Instruct']
new_model_names = ['gemma2-2b', 'phi3-4b', 'llama2-7b', 'phi3-7b', 'llama3-8b', 'llama3.1-8b',  'gemma2-9b',  'gemma2-27b', 'mixtral0.1-47b', 'llama3-70b', 'llama3.1-70b', 'qwen2-72b']


def plot_reliability_diagram(all_confidences, all_correctness, num_bins=10, save_path=None, display_percentages=True, success_percentage=1):
    """
    Create a reliability diagram illustrating the calibration of a model.

    Parameters
    ----------
    all_confidences: List[float]
        List of all the confidence scores ona given split.
    all_correctness: List[int]
        List of all the results of answers of the target LLM on a split, as zeros and ones for correctness.
    num_bins: int
        Number of bins used for plotting. Defaults to 10.
    save_path: Optional[str]
        Path to save the plot to. If None, the plot is shown directly.
    display_percentages: bool
        Indicate whether bins should include the percentage of points falling into them. Defaults to True.
    success_percentage: float
        The percentage of successful confidence scores. Defaults to 1, but if is not will be printed in a box on the
        bottom left.
    """

    if save_path == None:
        assert 'Please provide a save path'
    bins = np.arange(0.0, 1.0, 1.0 / num_bins)
    bins_per_prediction = np.digitize(all_confidences, bins)
    df = pd.DataFrame(
        {
            "y_pred": all_confidences,
            "y": all_correctness,
            "pred_bins": bins_per_prediction,
        }
    )

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean y and predicted probabilities per bin
    grouped_bins = grouped_by_bins.mean()
    grouped_bins = grouped_bins["y"].reindex(range(1, num_bins + 1), fill_value=0)
    bin_values = grouped_bins.values

    # calculate the number of items per bin
    bin_sizes = grouped_by_bins["y"].count()
    bin_sizes = bin_sizes.reindex(range(1, num_bins + 1), fill_value=0)

    plt.figure(figsize=(4, 4), dpi=200)
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")
    step_size = 1.0 / num_bins

    # Get bar colors
    bar_colors = []

    # Display the amount of points that fall into each bin via different shading
    if display_percentages:
        total = sum(bin_sizes.values)

        for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
            bin_percentage = bin_size / total * success_percentage
            cmap = matplotlib.colormaps.get_cmap("Blues")
            bar_colors.append(cmap(min(0.9999, bin_percentage + 0.2)))

    plt.bar(
        bins + step_size / 2,
        bin_values,
        width=0.09,
        alpha=0.8,
        color=bar_colors,  # "royalblue",
        edgecolor="black",
    )
    plt.plot(
        np.arange(0, 1 + 0.05, 0.05),
        np.arange(0, 1 + 0.05, 0.05),
        color="black",
        alpha=0.4,
        linestyle="--",
    )

    # Now add the percentage value of points per bin as text
    if display_percentages:
        total = sum(bin_sizes.values)
        eps = 0.01

        for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
            bin_percentage = round(bin_size / total * success_percentage * 100, 2)

            # Omit labelling for very small bars
            if bin_size == 0 or bin_values[i] < 0.2:
                continue

            plt.annotate(
                f"{bin_percentage} %",
                xy=(bin + step_size / 2, bin_values[i] - eps),
                ha="center",
                va="top",
                rotation=90,
                color="white" if bin_percentage > 40 else "black",
                alpha=0.7 if bin_percentage > 40 else 0.8,
                fontsize=10,
            )

        # Display success percentage if it is not 1
        if success_percentage < 1:
            plt.annotate(
                f"Success: {round(success_percentage * 100, 2)} %",
                xy=(-0.19, -0.135),
                color="royalblue",
                fontsize=10,
                alpha=0.7,
                annotation_clip=False,
                bbox=dict(facecolor="none", edgecolor="royalblue", pad=4.0, alpha=0.7),
            )

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Confidence", fontsize=14, alpha=0.8)
    plt.ylabel("Accuracy", fontsize=14, alpha=0.8)
    plt.tight_layout()

    if save_path is None:
        plt.show()

    else:
        plt.savefig(
            
        )

def save_plot(bin_accuracies, midpoints, bin_size, save_path=None):

    if save_path == None:
        assert 'Please provide a save path'


    num_bins = 15
    # r1 = mu + sigma * np.random.randn(437)
    sug_spec = np.arange(0, 1, 0.01)

    bins = bin_size

    fig, ax = plt.subplots()

    ax.set_xlabel('Confidence bin')

    # Plot lines
    color = 'tab:grey'
    ax.plot(sug_spec, sug_spec, color=color, linestyle='--')

    ax.patch.set_visible(False)

    ax.set_ylim(0, 1.0)
    # ax2.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    # ax2.set_xlim(0, 1.0)

    ax.bar(midpoints, bin_accuracies, 1/num_bins, edgecolor="k")


    plt.savefig(save_path, format="pdf", bbox_inches="tight")



def plot_accuracy_bar_diagram():

    df_small_models = pd.read_csv('eval_results/acc_success_small.csv', encoding='utf-8')
    df_large_models = pd.read_csv('eval_results/acc_success_large.csv', encoding='utf-8')

    df_all = pd.concat([df_small_models, df_large_models])
    # models = df_all['model'].unique().tolist()
    # print(models)
    datanames = [ 'triviaqa', 'sciq', 'wikiqa', 'nq' ]
    new_datanames = [ 'TriviaQA', 'Sciq', 'WikiQA', 'NQ' ]
    prompt_types = ['vanilla', 'guess', 'cot', 'demo'  ]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)


    for idx, (dataname, ax) in enumerate(zip(datanames, [ax1, ax2, ax3, ax4])):
        

        data_df = pd.DataFrame(columns=['prompt_type'].extend(new_model_names))
        data_df['prompt_type'] = prompt_types
        for model, new_model in zip(old_model_names, new_model_names):
            df_model_tmp = df_all[df_all['model']==model]
            print(df_model_tmp)
            data_df[new_model] = df_model_tmp[dataname+'_acc'].tolist()
            # print(model_acces)
            # sort_model_acces = [model_acces[3], model_acces[2], model_acces[0], model_acces[1]]
            # print(sort_model_acces)


        # Plot a bar chart using the DF
        data_df.plot(kind="bar", ax=ax)

        ax.set_xlabel(new_datanames[idx])
        ax.set_xticks(list(range(len(prompt_types))), ['Verb.', 'Zero-shot', 'CoT', 'Few-shot'], rotation=360)
        ax.set_ylim([15, 80])

        if idx == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_yticks([])
        if idx < 3:    
            ax.legend_ = None
        else:
           
            ax.legend(loc="upper right", ncol=6, bbox_to_anchor=(0.9, 1.25))
            
    # Change the plot dimensions (width, height)
    fig.set_size_inches(13, 2.8)
  
    # Export the plot as a PNG file
    plt.subplots_adjust(wspace=0.02, hspace=0)
    plt.savefig("model_accuracy_barplot.pdf",format="pdf", bbox_inches="tight")

if __name__ == '__main__':

    plot_accuracy_bar_diagram()