library(compiler)
enableJIT(3)
library("tmca.classify")

# For parallelization: register backends
if(.Platform$OS.type == "unix") {
  require(doMC)
  registerDoMC(32)
} else {
  require(doParallel)
  workers <- makeCluster(4, type="SOCK")
  registerDoParallel(workers)
}

# Manifesto data example
# ----------------------
initial_training_size <- 200
codeSetForClassification <- c("504", "411", "501", "506", "605", "303", "706", "301", "107", "402")

current_class <- "411"

# read manifesto data
data("manifestos")

# prepare data for LDA features
pseudoc_length <- 25
n <- nrow(manifesto_data)
pseudo_docs_idx <- rep(1:ceiling(n / pseudoc_length), each = pseudoc_length, length.out = n)
manifesto_pseudo_docs <- aggregate(manifesto_data$content, by = list(pseudo_doc = pseudo_docs_idx), paste, collapse = " ")

# select code for binary classification
manifesto_data$category <- ifelse(manifesto_data$cmp_code == current_class, current_class, "Other")
manifesto_data$category <- factor(manifesto_data$category, levels = c("Other", current_class))
manifestos <- factor(paste(manifesto_data$country, manifesto_data$party, manifesto_data$year, sep = "_"))


load("ldaFeaturesAll.RData")



# strategies <- c("random", "LC", "MC", "LCB", "LCBMC")
strategies <- c("random", "LC", "LCB", "LCBMC")
# strategies <- c("random", "LC")

write("", "progress.txt")
nALrunsForStatisticalTest <- 25
nRuns <- nALrunsForStatisticalTest * length(strategies) * length(codeSetForClassification)
currentRun <- 0

all_experiments <- data.frame()
for (strategy in strategies) {
  for (activeLearningRun in 1:nALrunsForStatisticalTest) {
    for (codeSet in 1:length(codeSetForClassification)) {
      new_exp <- data.frame(strategy, activeLearningRun, codeSet)
      all_experiments <- rbind(all_experiments, new_exp)
    }
  }
}


AlResultList_N_Iterations_List <- foreach (exp_i = 1:nrow(all_experiments)) %dopar% {
  current_exp <- all_experiments[exp_i, ]
  strategy <- current_exp$strategy
  activeLearningRun <- current_exp$activeLearningRun
  codeSet <- current_exp$codeSet
  
  message <- paste0("STRATEGY: ", strategy, " AL_RUN: ", activeLearningRun, " CODE:", codeSet, " RUN: ", exp_i, " OF ", nrow(all_experiments))
  write(message, "progress.txt", append = T)
  
  ###
  current_class <- codeSetForClassification[codeSet]
  manifesto_data$category <- ifelse(manifesto_data$cmp_code == current_class, current_class, "Other")
  manifesto_data$category <- factor(manifesto_data$category, levels = c("Other", current_class))
  experiment <- tmca_classify(corpus = manifesto_data$content, gold_labels = manifesto_data$category, extract_ngrams = F)
  experiment$dfm_ngram <- FullMatrix[, 1:(ncol(FullMatrix) - 50)]
  experiment$dfm_lda <- FullMatrix[, (ncol(FullMatrix) - 49):ncol(FullMatrix)]
  experiment$reset_active_learning()
  experiment$set_validation_AL_corpus()
  experiment$create_initial_trainingset(initial_training_size)
  experiment$active_learning(stop_threshold = 0.99, positive_class = current_class, strategy = strategy, facets = manifestos, max_iterations = 200, stop_window = 200)
  
  current_exp_result <- data.frame(
    strategy = strategy,
    codeSet = codeSet,
    activeLearningRun = activeLearningRun,
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
allResultsForComparison <- do.call(rbind, AlResultList_N_Iterations_List)


save(allResultsForComparison, file = "5_A0_Query_Comparison_FINAL_LCB-corrected.RData")
# load("5_A0_Query_Comparison_FINAL.RData")
# load("5_A0_Query_Comparison_FINAL_LCB-corrected.RData")



require(ggplot2)
require(reshape2)
category_names <- codeSetForClassification

df_strategies <- NULL
for (strategy in strategies) {
  df <- allResultsForComparison[allResultsForComparison$strategy == strategy, ]
  df <- aggregate(cbind(kappa, corProp, RMSD, PosEx, Certainty) ~ codeSet + iter, df, mean)
  df$codeSet <- as.factor(df$codeSet)
  levels(df$codeSet) <- category_names
  df$iter <- as.factor(df$iter)
  print(ggplot(df, aes(x = iter, y = corProp)) +
          geom_line(aes(group = codeSet, color = codeSet), size = 1) +
          ylab("Pearson's r") +
          ylim(0, 1) +
          ggtitle(strategy) )
  df_strategies <- rbind(df_strategies, cbind(strategy = strategy, df))
}


df_strategies$strategy <- as.factor(df_strategies$strategy)
levels(df_strategies$strategy) <- c("RND", "LC", "LCB", "LCBMC")


source("multiplot.R")

# FINAL PLOT FOR PAPER: r
dodge <- position_dodge(2)
df_for_paper <- df_strategies[df_strategies$strategy != "LCBMC", ]

p1 <- ggplot(df_for_paper, aes(iter, corProp, group = strategy, color = strategy)) +
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(data = df_for_paper[as.integer(df_for_paper$iter) %% 10 == 0, ], 
               aes(iter, corProp, group = strategy, color = strategy), 
               fun.data = mean_se, geom = "errorbar") +
  ylab("Pearson's r") + ylim(0.6, 1) + scale_color_grey() + theme_bw() +
  scale_x_discrete(breaks = levels(pd$iter)[c(rep(F, 19), T)], name = "Iteration")


# EXTRA PLOT: kappa
p2 <- ggplot(df_for_paper, aes(iter, kappa, group = strategy, color = strategy)) +
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(data = df_for_paper[as.integer(df_for_paper$iter) %% 10 == 0, ], 
               aes(iter, kappa, group = strategy, color = strategy), 
               fun.data = mean_se, geom = "errorbar", position = dodge) +
  ylab("kappa") + scale_color_grey() + theme_bw() +
  scale_x_discrete(breaks = levels(pd$iter)[c(rep(F, 19), T)], name = "Iteration")

# EXTRA PLOT: RMSD
p3<- ggplot(df_for_paper, aes(iter, RMSD, group = strategy, color = strategy)) +
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(data = df_for_paper[as.integer(df_for_paper$iter) %% 10 == 0, ], 
               aes(iter, RMSD, group = strategy, color = strategy), 
               fun.data = mean_se, geom = "errorbar", position = dodge) +
  ylab("RMSD") + scale_color_grey() + theme_bw() +
  scale_x_discrete(breaks = levels(pd$iter)[c(rep(F, 19), T)], name = "Iteration")

# LEFT OUT PLOT: Positive examples
p4 <- ggplot(df_for_paper, aes(iter, PosEx, group = strategy, color = strategy)) +
  stat_summary(fun.y = mean, geom = "line") + 
  stat_summary(data = df_for_paper[as.integer(df_for_paper$iter) %% 10 == 0, ], 
               aes(iter, PosEx, group = strategy, color = strategy), 
               fun.data = mean_se, geom = "errorbar", position = dodge) +
  ylab("positive examples") + scale_color_grey() + theme_bw() +
  scale_x_discrete(breaks = levels(pd$iter)[c(rep(F, 19), T)], name = "Iteration")


multiplot(p1, p2, p3, p4, cols = 2)


# LEFT OUT PLOT: Stability
ggplot(df_for_paper, aes(iter, Certainty, group = strategy, color = strategy)) +
  stat_summary(fun.y = mean, geom = "line") + ylim(0.9, 1) +
  stat_summary(data = df_for_paper[as.integer(df_for_paper$iter) %% 10 == 0, ], 
               aes(iter, Certainty, group = strategy, color = strategy), 
               fun.data = mean_se, geom = "errorbar", position = dodge) +
  ylab("Stability") + # scale_color_grey() +
  scale_x_discrete(breaks = levels(pd$iter)[c(rep(F, 9), T)], name = "Iteration") +
  geom_hline(yintercept = 0.99)




