import pandas as pd
import matplotlib.pyplot as plt

def plot_bars(file1, file2, file3, dataset, lang):
    # Load the results files into dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Combine all dataframes into a list for easier processing
    dataframes = [df1, df2, df3]
    models = ['Claude', 'GPT4o', 'Mistral']

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
        ax.bar(x, regular_means, width, label=f"Regular {metric_name}", color="skyblue")
        ax.bar([p + width for p in x], leap_means, width, label=f"LEAP {metric_name}", color="steelblue")

        # Set plot details
        ax.set_xlabel('Models')
        ax.set_ylabel(f'Average {metric_name}')
        ax.set_title(f'Average {metric_name} ({dataset} - {lang})')
        ax.set_xticks([p + width/2 for p in x])
        ax.set_xticklabels(models)
        if metric_name == "JTC_Score":
            ax.set_ylim(0, 1)
        elif metric_name == "Jaccard_Score":
            ax.set_ylim(0, 1)
        else:  
            ax.set_ylim(0, 5.5)
        ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.savefig(f'barplot_{metric_name}_{dataset}_{lang}.png')


def plot_histogram(csv_file, model, lang, dataset):
    # Load the data from the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the relevant columns
    regular_scores = df["Regular_JTC_Score"]
    leap_scores = df["LEAP_JTC_Score"]
    
    # Plot overlapping histograms
    plt.figure(figsize=(8, 6))
    plt.hist(regular_scores, bins=50, alpha=0.6, label="Regular JTC Scores", color="blue")
    plt.hist(leap_scores, bins=50, alpha=0.6, label="LEAP JTC Scores", color="red")
    
    # Set plot details
    plt.title(f"{model} Regular and LEAP JTC Scores ({dataset} - {lang})")
    plt.xlabel("JTC Scores")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(f'histogram_{dataset}_{lang}_{model}.png')

plot_bars('../results/claude_French_0_results.csv', '../results/gpt4o_French_0_results.csv', '../results/mistral_French_0_results.csv', 'Law', 'French')
plot_bars('../results/claude_French_1_results.csv', '../results/gpt4o_French_1_results.csv', '../results/mistral_French_1_results.csv', 'Medical', 'French')
plot_bars('../results/claude_Simplified_Chinese_0_results.csv', '../results/gpt4o_Simplified_Chinese_0_results.csv', '../results/mistral_Simplified_Chinese_0_results.csv', 'Law', 'Simplified Chinese')
plot_bars('../results/claude_Simplified_Chinese_1_results.csv', '../results/gpt4o_Simplified_Chinese_1_results.csv', '../results/mistral_Simplified_Chinese_1_results.csv', 'Medical', 'Simplified Chinese')
# plot_histogram('../results/claude_French_0_results.csv', 'Claude', 'French', 'Law')
# plot_histogram('../results/claude_French_1_results.csv', 'Claude', 'French', 'Medical')
# plot_histogram('../results/gpt4o_French_0_results.csv', 'GPT4o', 'French', 'Law')
# plot_histogram('../results/gpt4o_French_1_results.csv', 'GPT4o', 'French', 'Medical')
# plot_histogram('../results/mistral_French_0_results.csv', 'Mistral', 'French', 'Law')
# plot_histogram('../results/mistral_French_1_results.csv', 'Mistral', 'French', 'Medical')
# plot_histogram('../results/claude_Simplified_Chinese_0_results.csv', 'Claude', 'Simplified Chinese', 'Law')
# plot_histogram('../results/claude_Simplified_Chinese_1_results.csv', 'Claude', 'Simplified Chinese', 'Medical')
# plot_histogram('../results/gpt4o_Simplified_Chinese_0_results.csv', 'GPT4o', 'Simplified Chinese', 'Law')
# plot_histogram('../results/gpt4o_Simplified_Chinese_1_results.csv', 'GPT4o', 'Simplified Chinese', 'Medical')
# plot_histogram('../results/mistral_Simplified_Chinese_0_results.csv', 'Mistral', 'Simplified Chinese', 'Law')
# plot_histogram('../results/mistral_Simplified_Chinese_1_results.csv', 'Mistral', 'Simplified Chinese', 'Medical')