# Proportional classification
# ===========================

# We test the readme package for accuracy of predicting proportions
# in arbitrary data subsets by using one (large) training set, and
# test sets with different compositional properties:
# - temporal splits by year
# - split by country
# - split by party
# - split by manifestor
# - random splits
# - odd/even test set split
# The results show that HK2010 as implemented in the ReadMe package
# fails to predict proportions correctly, when feature distributions
# in the test set largely vary from the training set.
# When feature distrbutions are similar, as for the random split set
# and the odd/even split, proportions are (very) accurate.
# In conclusion: HK2010 should not be used to compare proportortions
# cross-sectional or for diachronic data.


options(stringsAsFactors = F)
rmsd <- function(y1, y2) {
  if ((length(y1) != length(y2)) | length(y1) == 0) 
    stop("length of arguments does not match / is not greater than 0")
  keepIdx <- !is.na(y1)
  y1 <- y1[keepIdx]
  y2 <- y2[keepIdx]
  result <- sqrt(sum((y1 - y2) ^ 2) / length(y1))
  return(result)
}

# STEP 1. Split Test Data
# -------------------
library("tmca.classify")
data(manifestos)


manifesto_data$countryParty <- paste0(manifesto_data$country, "_", manifesto_data$party)
manifesto_data$manifesto <- paste(manifesto_data$year, manifesto_data$country, manifesto_data$party, sep = "_")
set.seed(9721) # fix random seed
manifesto_data$randomSplit <- paste0("RND", sample(1:10, nrow(manifesto_data), replace = T))

# create test set splits
split_year <- lapply(sort(unique(manifesto_data$year)), FUN = function(x) list(name = x, type = "YEAR", rowIds = which(manifesto_data$year == x)))
split_country <- lapply(sort(unique(manifesto_data$country)), FUN = function(x) list(name = x, type = "COUNTRY", rowIds = which(manifesto_data$country == x)))
split_party <- lapply(sort(unique(manifesto_data$countryParty)), FUN = function(x) list(name = x, type = "PARTY", rowIds = which(manifesto_data$countryParty == x)))
split_manifesto <- lapply(sort(unique(manifesto_data$manifesto)), FUN = function(x) list(name = x, type = "MANIFESTO", rowIds = which(manifesto_data$manifesto == x)))
split_random <- lapply(sort(unique(manifesto_data$randomSplit)), FUN = function(x) list(name = x, type = "RANDOM", rowIds = which(manifesto_data$randomSplit == x)))
split_all <- list(list(name = "ALL", type = "ALL", rowIds = 1:nrow(manifesto_data)))

dataSetSplits <- c(split_all, split_random, split_manifesto, split_year, split_country, split_party)
length(dataSetSplits)

# If not installed, do install ReadMe package from github
# ------
# library(devtools)
# install_github("iqss-research/VA-package")
# install_github("iqss-research/ReadMeV1", force = T) 

library(ReadMe)
# I needed to change the python call of the undergrad function, to work on my windows machine
# On other systems, the package function might work out of the box
source("undergrad.R") 

# Switch to the data directory
setwd("readme_data")


# PREPARE DATA
prepare_data_for_readme <- TRUE
# This block writes all coded manifesto sentences into a data
# structure for readme (> 44,000 files). Every sentence/document 
# is written into a single text file. A file control.txt is 
# created to keep information about IDs and labels of training 
# and test data. This data structure is used by ReadMe.
# Caution: This may take a while ...
if (prepare_data_for_readme) {
  
  # Write out sentences as lines in single files
  i <- 0
  for (line in manifesto_data$content) {
    i <- i + 1 
    writeLines(line, paste0(i, ".txt"), useBytes = T)
  }
  
  # write control.txt
  selected_code <- "501"
  ROWID <- paste0(1:nrow(manifesto_data),".txt")
  TRUTH <- ifelse(manifesto_data$cmp_code == selected_code, 1, 0)
  TRAININGSET <- rep(c(1, 0), 1 + (nrow(manifesto_data) / 2), length.out = nrow(manifesto_data))
  df_control <- data.frame(ROWID, TRUTH, TRAININGSET)
  write.table(df_control, sep = ",", row.names = F, col.names = T, file = "control.txt", quote = F, eol = "\n")
  
  # on all test data
  manifesto_data_results <- undergrad(sep = ',', python3 = T, control = "control.txt")
  manifesto_data_preprocess <- preprocess(manifesto_data_results)
  manifesto_data_hk2010 <- readme(manifesto_data_preprocess)
  print(manifesto_data_hk2010$est.CSMF)
  print(manifesto_data_hk2010$true.CSMF)
}




# Define codes for tests
codeSetForClassification <- c("504", "411", "501", "506", "605", "303", "706", "301", "107", "402")

manifesto_results <- data.frame()

# Run readme for each code and the prepared experiment test sets (10 codes * 94 splits)
# Caution: This may run several hours
for (selected_code in codeSetForClassification) {
  
  for (i in 1:(length(dataSetSplits))) {
    
    print("==========================================")
    print(paste0("SPLIT NR ", i))
    print("==========================================")
    
    # Start with full training / test set for control.txt
    ROWID <- paste0(1:nrow(manifesto_data),".txt")
    TRUTH <- ifelse(manifesto_data$cmp_code == selected_code, 1, 0)
    TRAININGSET <- rep(c(1, 0), 1 + (nrow(manifesto_data) / 2), length.out = nrow(manifesto_data))
    df_control <- data.frame(ROWID, TRUTH, TRAININGSET, manifesto = manifesto_data$manifesto)
    # View(df_control[df_control$TRAININGSET == 0, ])
    
    # Remove items from the entire test set not part of the current test set
    split_test_ids <- dataSetSplits[[i]]$rowIds
    all_test_ids <- which(TRAININGSET == 0)
    test_ids_to_remove <- setdiff(all_test_ids, split_test_ids)
    if (length(test_ids_to_remove) > 0) {
      df_control <- df_control[-test_ids_to_remove, ]
    }
    # View(df_control[df_control$TRAININGSET == 0, ])
    
    # write readme controlfile containing complete training set and reduced test
    control_file_name <- paste0("control_", selected_code, "_", i, ".txt")
    write.table(df_control[, 1:3], sep = ",", row.names = F, col.names = T, file = control_file_name, quote = F, eol = "\n")
    
    # run readme on complete training set and reduced test
    manifesto_data_results <- undergrad(sep = ',', python3 = T, control = control_file_name)
    manifesto_data_preprocess <- preprocess(manifesto_data_results)
    manifesto_data_hk2010 <- readme(manifesto_data_preprocess)
    
    n <- length(dataSetSplits[[i]]$rowIds)
    split_result <- c(manifesto_data_hk2010$est.CSMF["1"], manifesto_data_hk2010$true.CSMF["1"])
    split_result <- c(selected_code, 
                      dataSetSplits[[i]]$type, 
                      dataSetSplits[[i]]$name, 
                      n, 
                      split_result)
    
    # Append results to result collection variable
    manifesto_results <- rbind(manifesto_results, split_result)
    
    # Just to monitor the progress: write tmp result
    write.csv(manifesto_results, file = paste0("ReadMe_tmp_result_", selected_code , ".csv"))
    
  }
}


print(manifesto_results)
View(manifesto_results)

colnames(manifesto_results) <- c("code", "type", "split", "n", "est", "truth")
manifesto_results$est <- as.numeric(manifesto_results$est)
manifesto_results$truth <- as.numeric(manifesto_results$truth)

save(manifesto_results, file = "ReadMe_test.Rdata")

# --------------------------------------------------
load("ReadMe_test.Rdata")


result_table <- data.frame()
result_table2 <- data.frame(code = unique(manifesto_results$code))
all_types <- unique(manifesto_results$type)
for (type in all_types) {
  type_res <- NULL
  for (code in unique(manifesto_results$code)) {
    sel_idx <- manifesto_results$code == code & manifesto_results$type == type
    rmsd_data <- manifesto_results[sel_idx, c("est", "truth")]
    rmsd_value <- rmsd(rmsd_data[, 1], rmsd_data[, 2])
    n <- nrow(rmsd_data)
    result_table <- rbind(result_table, data.frame(
      type = type,
      code = code,
      n = n,
      rmsd = rmsd_value
    ))
    type_res <- c(type_res, rmsd_value)
  }
  result_table2 <- cbind(result_table2, type_res)
}
colnames(result_table2) <- c("code", all_types)
View(result_table)
View(result_table2)







# plot poorly predicted vs true time line
code <- "504"
time_series <- manifesto_results[manifesto_results$code == code & manifesto_results$type == "YEAR", c("split", "est", "truth")]
matplot(time_series[, 1], time_series[, 2:3], type = "l", ylab = "proportion", xlab = "year")
legend("topleft", legend=c("estimation", "truth"), lty=1:2, col=1:2, text.col=1:2)
