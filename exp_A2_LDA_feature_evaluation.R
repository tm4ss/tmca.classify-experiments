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


all_experiments <- data.frame(
  codeSet = 1:10,
  use_lda_features = rep(c(TRUE, FALSE), length(codeSetForClassification)),
  n_repeat = rep(1:20, 10)
)

classification_result_list <- foreach (exp_i = 1:nrow(all_experiments)) %dopar% {
  codeSet <- all_experiments$codeSet[exp_i]
  n_repeat <- all_experiments$n_repeat[exp_i]
  use_lda <- all_experiments$use_lda_features[exp_i]
  
  current_class <- codeSetForClassification[codeSet]
  manifesto_data$category <- ifelse(manifesto_data$cmp_code == current_class, current_class, "Other")
  manifesto_data$category <- factor(manifesto_data$category, levels = c("Other", current_class))
  experiment <- tmca_classify(corpus = manifesto_data$content, gold_labels = manifesto_data$category, extract_ngrams = F)
  experiment$dfm_ngram <- FullMatrix[, 1:(ncol(FullMatrix) - 50)]
  
  if (use_lda) {
    experiment$dfm_lda <- FullMatrix[, (ncol(FullMatrix) - 49):ncol(FullMatrix)]
  }
  
  experiment$reset_active_learning()
  experiment$set_validation_AL_corpus()
  experiment$create_initial_trainingset(initial_training_size)
  experiment$active_learning(stop_threshold = 0.99, positive_class = current_class, strategy = "LCB", facets = manifestos, max_iterations = 200, stop_window = 200)
  
  current_exp_result <- data.frame(
    n_repeat = n_repeat,
    codeSet = codeSet,
    use_lda = use_lda,
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
  
  return(current_exp_result)
}
classification_result <- data.table::rbindlist(classification_result_list)

# save(classification_result, file = "exp_lda_result.RData")

source("multiplot.R")
require(ggplot2)
pd <- position_dodge(0.1) # move them .05 to the left and right
p1 <- ggplot(classification_result, aes(x = iter, y = corProp, group = use_lda, color = use_lda)) +
  theme_bw() + scale_color_grey() + theme(legend.position = "bottom") +
  stat_summary(fun.y = "mean", geom = "line") +
  stat_summary(data = classification_result[as.integer(classification_result$iter) %% 10 == 0, ],
               fun.data = mean_se, geom = "errorbar") +
  ylab("Pearson's r") + xlab("Iteration")
p2 <- ggplot(classification_result, aes(x = iter, y = kappa, group = use_lda, color = use_lda)) +
  theme_bw() + scale_color_grey() + theme(legend.position = "bottom") +
  stat_summary(fun.y = "mean", geom = "line") +
  stat_summary(data = classification_result[as.integer(classification_result$iter) %% 10 == 0, ],
               fun.data = mean_se, geom = "errorbar") +
  ylab("kappa") + xlab("Iteration")
multiplot(p1, p2, cols = 2)
