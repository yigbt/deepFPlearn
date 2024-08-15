# Load necessary libraries
library(ggplot2)

# Load the data
data <- read.csv("reversed_data.csv")

# Create the plot
ggplot(data, aes(x = AR, y = predicted)) +
  geom_point(color = 'blue', alpha = 0.6) +  # Adds points with some transparency
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = "dashed") +  # Adds the y=x reference line
  labs(title = "Comparison of AR values and Predicted values",
       x = "AR values (Real)",
       y = "Predicted values") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Save the plot as a PNG file
ggsave("ar_vs_predicted_comparison_plot.png")

