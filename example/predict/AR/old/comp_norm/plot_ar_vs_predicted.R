
# Load necessary libraries
library(ggplot2)

# Load the data
data <- read.csv("predict_data_AR.csv")

max_value <- max(c(data$AR, data$predicted))
# Create the plot
ggplot(data, aes(x = AR, y = predicted)) +
  geom_point(color = 'blue', alpha = 0.6) +  # Adds points with some transparency
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = "dashed") +  # Adds the y=x reference line
  labs(
       x = "Actual Values",
       y = "Predicted values") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlim(NA, 250) +  # Set x-axis limit
  ylim(NA, 250)  # Set y-axis limit
# Save the plot as a PNG file
ggsave("ar_comp.png")

