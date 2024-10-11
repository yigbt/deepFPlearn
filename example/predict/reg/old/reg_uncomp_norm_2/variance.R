# Load necessary library
library(readr)

# Load the dataset from the CSV file

df <- read_csv("predict_data_AR.csv", col_types = cols(
  ...1 = col_skip(),       # Skip the extra index column
  smiles = col_character(),
  AR = col_double(),
  predicted = col_double()
))

total_variance <- var(df$AR, na.rm = TRUE) * (length(df$AR) - 1)
residuals <- df$AR - df$predicted
residual_sum_of_squares <- sum(residuals^2, na.rm = TRUE)

r_squared <- 1 - (residual_sum_of_squares / total_variance)
cat("R²:", r_squared, "\n")
