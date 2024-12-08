import pandas as pd
import matplotlib.pyplot as plt

def plot_results(file1, file2, file3):
    # Load the results files into dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Combine all dataframes into a list for easier processing
    dataframes = [df1, df2, df3]
    models = ['Model 1', 'Model 2', 'Model 3']

    # Initialize lists to store the averages for each metric
    metrics = {
        "JTC_Score": ["Regular_JTC_Score", "LEAP_JTC_Score"],
        "Jaccard_Score": ["Regular_Jaccard_Score", "LEAP_Jaccard_Score"],
        "chrF": ["Regular_chrF", "LEAP_chrF"]
    }

    # Plotting
    for metric_name, columns in metrics.items():
        regular_means = []
        leap_means = []
        
        for df in dataframes:
            regular_means.append(df[columns[0]].mean())
            leap_means.append(df[columns[1]].mean())

        # Create bar graph
        x = range(len(models))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x, regular_means, width, label=f"Regular {metric_name}")
        ax.bar([p + width for p in x], leap_means, width, label=f"LEAP {metric_name}")

        # Set plot details
        ax.set_xlabel('Models')
        ax.set_ylabel(f'Average {metric_name}')
        ax.set_title(f'Average {metric_name} Comparison Across Models')
        ax.set_xticks([p + width/2 for p in x])
        ax.set_xticklabels(models)
        ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

# Example usage
plot_results('../results/claude_French_0_results.csv', '../results/gpt4o_French_0_results.csv', '../results/mistral_French_0_results.csv')
