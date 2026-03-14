import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_percentile_approach():
    # 1. Generate Mock Data (Simulating 1558 players)
    # real game data usually has a "long tail" (Gamma distribution)
    # Most players have 0.5 - 1.2 KD, a few have 3.0+
    np.random.seed(42)
    data = np.random.gamma(shape=2.0, scale=0.5, size=1558)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['kdRatio'])

    # 2. Calculate the Percentile Thresholds
    # We want the Bottom 20% and the Top 20%
    lower_threshold = df['kdRatio'].quantile(0.20)  # 20th Percentile
    upper_threshold = df['kdRatio'].quantile(0.80)  # 80th Percentile

    print(f"Low Skill Cutoff (Bottom 20%): KD < {lower_threshold:.2f}")
    print(f"High Skill Cutoff (Top 20%): KD > {upper_threshold:.2f}")

    # 3. Plotting the Graph
    plt.figure(figsize=(10, 6))

    # Plot the main distribution histogram
    # 'bins=50' splits the data into 50 bars
    # 'alpha=0.6' makes it slightly transparent
    n, bins, patches = plt.hist(df['kdRatio'], bins=50, color='lightgray', edgecolor='black', alpha=0.6)

    # 4. Color the Regions
    # We iterate through the bars (patches) and color them based on where they sit
    for i in range(len(patches)):
        # Calculate the center of the bar
        bar_center = (bins[i] + bins[i + 1]) / 2

        if bar_center < lower_threshold:
            patches[i].set_facecolor('#ff9999')  # Red for Low Skill
        elif bar_center > upper_threshold:
            patches[i].set_facecolor('#99ff99')  # Green for High Skill
        else:
            patches[i].set_facecolor('#99ccff')  # Blue for Medium Skill

    # 5. Add Vertical Lines for Clarity
    plt.axvline(lower_threshold, color='red', linestyle='dashed', linewidth=2,
                label=f'20th Percentile ({lower_threshold:.2f})')
    plt.axvline(upper_threshold, color='green', linestyle='dashed', linewidth=2,
                label=f'80th Percentile ({upper_threshold:.2f})')

    # Labels and Titles
    plt.title('Approach 3: Percentile-Based Labeling (Data Distribution)', fontsize=14)
    plt.xlabel('K/D Ratio', fontsize=12)
    plt.ylabel('Number of Players', fontsize=12)

    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', edgecolor='black', label='Low Skill (Bottom 20%)'),
        Patch(facecolor='#99ccff', edgecolor='black', label='Medium Skill (Middle 60%)'),
        Patch(facecolor='#99ff99', edgecolor='black', label='High Skill (Top 20%)')
    ]
    plt.legend(handles=legend_elements)

    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    plot_percentile_approach()