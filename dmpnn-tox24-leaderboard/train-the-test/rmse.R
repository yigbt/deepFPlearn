library(readr)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]

data <- read_csv(input_file)

actual_values <- data$activity_scaled
predicted_values <- data$predicted

rmse <- sqrt(mean((actual_values - predicted_values)^2))

# Step 4: Print the RMSE value
print(paste("The RMSE is:", rmse))

