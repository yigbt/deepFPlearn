import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


# Function to find the CSV file with the lowest val_binary_cross_entropy
def find_best_csv(directory):
    best_csv_path = None
    lowest_loss = float("inf")

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("scores") and file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                data = pd.read_csv(csv_path)
                loss_data = data[data["metric"] == "binary_cross_entropy"]
                val_loss = loss_data[loss_data["set_"] == "validation"]["score"].iloc[
                    -1
                ]

                if val_loss < lowest_loss:
                    lowest_loss = val_loss
                    best_csv_path = csv_path

    return best_csv_path


# Directories containing subfolders with CSV files
directories = [
    "dmpnn-weight-AR",
    "dmpnn-weight-ER",
    "dmpnn-weight-ED",
    "dmpnn-random-AR",
    "dmpnn-random-ER",
    "dmpnn-random-ED",
]
metrics = ["f1", "mcc", "binary_cross_entropy"]
# Calculate the number of rows and columns for the subplot grid
num_directories = len(directories)
num_cols = len(metrics)
unique_row_values = list(set([directory.split("-")[-2] for directory in directories]))
num_rows = len(unique_row_values)

# Create a color map
cmap = cm.get_cmap("inferno")

# Create a figure with AxB grid for the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8), sharex=True, sharey=True)

# Iterate over the metrics
for i, metric in enumerate(metrics):
    # Iterate over the directories
    for j, directory in enumerate(directories):
        # Extract the target and row value from the directory name
        target = directory.split("-")[-1]
        row_value = directory.split("-")[-2]

        # Find the best CSV file based on lowest val_binary_cross_entropy
        best_csv_path = find_best_csv(directory)

        if best_csv_path is not None:
            # Load the data from the best CSV file
            data = pd.read_csv(best_csv_path)

            # Filter the DataFrame for the current metric, target, and set
            filtered_data = data[
                (data["metric"] == metric)
                & (data["set_"].isin(["training", "validation"]))
            ]

            # Find the row index based on row value
            row_index = unique_row_values.index(row_value)

            # Select the subplot for the current metric and row value
            ax = axes[row_index, i]

            # Plot the lines for training and validation
            training_data = filtered_data[filtered_data["set_"] == "training"]
            validation_data = filtered_data[filtered_data["set_"] == "validation"]

            # Set color based on the target
            color = (
                cmap(0.1)
                if target == "AR"
                else cmap(0.4)
                if target == "ER"
                else cmap(0.9)
            )

            ax.plot(
                training_data["epoch"],
                training_data["score"],
                color=color,
                linestyle="-",
                label=f"Training ({target})",
            )
            ax.plot(
                validation_data["epoch"],
                validation_data["score"],
                color=color,
                linestyle="--",
                label=f"Validation ({target})",
            )

            ax.set_title(f"{metric}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()
            if i == 2:
                ax2 = ax.twinx()
                ax2.set_ylabel(
                    row_value,
                    rotation=-90,
                    va="bottom",
                    bbox=dict(
                        boxstyle="round", pad=0.2, facecolor="white", edgecolor="black"
                    ),
                )
                ax2.set_yticklabels([])
                ax2.yaxis.set_ticks_position("none")
            else:
                ax.set_yticklabels([])  # Hide y-label for other columns
                # if num_directories < num_rows * num_cols:
    for j in range(num_directories, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.savefig("subplots.png")
