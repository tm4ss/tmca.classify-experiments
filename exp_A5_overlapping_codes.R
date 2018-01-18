library(compiler)
enableJIT(3)
library("tmca.classify")

# For parallelization: register backends
if(.Platform$OS.type == "unix") {
  require(doMC)
  registerDoMC(4)
} else {
  require(doParallel)
  workers <- makeCluster(4, type="SOCK")
  registerDoParallel(workers)
}


# Manifesto data example
# ----------------------
initial_training_size <- 200
codeSetForClassification <- c("504", "411", "501", "506", "605", "303", "706", "301", "107", "402")

# read manifesto data
data("manifestos")
load("ldaFeaturesAll.RData")
manifestos <- factor(paste(manifesto_data$country, manifesto_data$party, manifesto_data$year, sep = "_"))


classification_result_list <- foreach (codeSet = codeSetForClassification) %dopar% {
  
  current_class <- codeSet
  manifesto_data$category <- ifelse(manifesto_data$cmp_code == current_class, current_class, "Other")
  manifesto_data$category <- factor(manifesto_data$category, levels = c("Other", current_class))
  experiment <- tmca_classify(corpus = manifesto_data$content, gold_labels = manifesto_data$category, extract_ngrams = F)
  experiment$dfm_ngram <- FullMatrix[, 1:(ncol(FullMatrix) - 50)]
  experiment$dfm_lda <- FullMatrix[, (ncol(FullMatrix) - 49):ncol(FullMatrix)]
  experiment$reset_active_learning()
  experiment$set_validation_AL_corpus()
  experiment$create_initial_trainingset(initial_training_size)
  experiment$active_learning(stop_threshold = 0.99, positive_class = current_class, strategy = "LCB", facets = manifestos, max_iterations = 200, stop_window = 3)
  
  current_exp_result <- data.frame(
    codeSet = codeSet,
    iter = experiment$progress$iteration,
    specificity = experiment$progress_validation$S,
    kappa = experiment$progress_validation$kappa,
    FTEST = experiment$progress_validation$F,
    corCount = experiment$progress_validation$r,
    corProp = experiment$progress_validation$r,
    RMSD = experiment$progress_validation$rmsd,
    PosEx = experiment$progress$n_pos,
    NegEx = experiment$progress$n_pos,
    Certainty = experiment$progress$stabililty
  )
  
  res <- list(model = experiment, evaluation = current_exp_result)
  
  return(res)
}

all_classification_results <- NULL
for (i in 1:length(codeSetForClassification)) {
  class_column <- classification_result_list[[i]]$model$classify()
  class_column <- ifelse(class_column == codeSetForClassification[i], T, F)
  all_classification_results <- cbind(all_classification_results, class_column)
}
colnames(all_classification_results) <- codeSetForClassification
View(all_classification_results)
rowSums(all_classification_results)


# plot histogram of code frequencies per sentence
require(ggplot2)
require(reshape2)
class_frequency <- melt(rowSums(all_classification_results))
n_bins <- length(unique(class_frequency$value))
ggplot(class_frequency, aes(value)) + 
  geom_histogram(bins = n_bins, col = "grey10", fill = "grey50") +
  scale_x_continuous(name="Codes per sentence", breaks = 0:n_bins) +
  theme_bw() + ylab("Sentences")


# which codes do overlap?
sentences_with_two_codes <- which(class_frequency$value == 2)
set.seed(1001)
sampled_ids <- sort(sample(sentences_with_two_codes, 5))
ml_codes <- NULL
for (i in sampled_ids) {
  ml_codes <- c(ml_codes, paste(names(which(all_classification_results[i, ])), collapse = ", "))
}
sample_df <- data.frame(
  id = sampled_ids,
  orig_code = manifesto_data$cmp_code[sampled_ids],
  ml_codes = ml_codes,
  content = manifesto_data$content[sampled_ids]
)
View(sample_df)


# count overlapping codes
all_code_pairs <- NULL
for (i in sentences_with_two_codes) {
  all_code_pairs <- c(all_code_pairs, paste(names(which(all_classification_results[i, ])), collapse = ", "))
}
sort(table(all_code_pairs))

# select specific example for 504/706 overlap
manifesto_data[14257, ]
