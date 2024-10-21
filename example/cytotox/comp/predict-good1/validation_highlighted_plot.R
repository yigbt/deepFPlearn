library(ggplot2)

# Load both datasets
data <- read.csv("example/cytotox/predict-good1/predict_data_AR.csv")  # Actual and predicted values
denormalized_data <- read.csv("example/cytotox/predict-good1/denormalized_output.csv")  # Denormalized file

# Round AR and denormalized values to 5 decimal places for comparison
data$AR_rounded <- round(data$AR, 5)
denormalized_data$denormalized_rounded <- round(denormalized_data$denormalized, 5)

# Merge the data based on the rounded AR and denormalized values
# The matched rows will be colored differently
data$color <- ifelse(data$AR_rounded %in% denormalized_data$denormalized_rounded, "yellow", "blue")

# Max value for x and y axis limit
max_value <- max(c(data$AR, data$predicted))

# Create the plot
ggplot(data, aes(x = AR, y = predicted)) +
  geom_point(aes(color = color), alpha = 0.6) +  # Color points based on matching
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = "dashed") +  # y=x line
  labs(
    x = "Actual Values",
    y = "Predicted values"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlim(-2, 3) +  # Set x-axis limit
  ylim(-2, 3) +  # Set y-axis limit
  scale_color_identity()  # Use the color set in the 'color' column directly

# Save the plot
ggsave("example/cytotox/predict-good1/ar_comp_validation.png")

