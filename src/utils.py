import matplotlib.pyplot as plt


def plot_boxplot(dataframes):
    fig, ax = plt.subplots()
    ax.boxplot(dataframes)
    ax.set_xticklabels(range(1, len(dataframes) + 1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Results")
    ax.set_title("Comparison of Regression Models")
    plt.show()
