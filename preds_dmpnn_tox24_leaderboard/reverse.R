library(readr)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1] 
output_file <- "dmpnn_test_tox24__leaderboard_predictions.csv"

data <- read_csv(input_file)

min_activity <- -45.00
max_activity <- 111.12
data$activity <- data$activity * (max_activity - min_activity) + min_activity
head(data)
write_csv(data, output_file)
