library(dplyr)

file1 <- read.csv("tox24_train_plus_leaderboard_scaled.csv")
file2 <- read.csv("DMPNN_train_scaled.csv")

selected_columns <- data.frame(file1[, c( "smiles","activity_scaled")], predictions = file2$predictions)

write.csv(selected_columns, "combined_file.csv", row.names = FALSE)
