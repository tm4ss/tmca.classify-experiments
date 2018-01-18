library(compiler)
enableJIT(3)
library("tmca.classify")

# For parallelization: register backends
if(.Platform$OS.type == "unix") {
  require(doMC)
  registerDoMC(44)
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

write("", "progress.txt")
nALrunsForStatisticalTest <- 100
initial_sampling_strategies <- c("RANDOM", "BIAS_TIME", "BIAS_COUNTRY", "BIAS_PARTY")
nRuns <- length(initial_sampling_strategies) * nALrunsForStatisticalTest * length(codeSetForClassification)

all_experiments <- data.frame()
for (initial_sampling_strategy in initial_sampling_strategies) {
  for (activeLearningRun in 1:nALrunsForStatisticalTest) {
    for (codeSet in 1:length(codeSetForClassification)) {
      new_exp <- data.frame(initial_sampling_strategy, activeLearningRun, codeSet)
      all_experiments <- rbind(all_experiments, new_exp)
    }
  }
}



AlResultList_N_Iterations_List <- foreach (exp_i = 1:nrow(all_experiments)) %dopar% {
  current_exp <- all_experiments[exp_i, ]
  initial_sampling_strategy <- current_exp$initial_sampling_strategy
  activeLearningRun <- current_exp$activeLearningRun
  codeSet <- current_exp$codeSet
  
  message <- paste0("STRATEGY: ", initial_sampling_strategy, " AL_RUN: ", activeLearningRun, " CODE:", codeSet, " RUN: ", exp_i, " OF ", nrow(all_experiments))
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
  
  ##### initial biased sampling start
  # SAMPLE INITIAL TRAINING SET
  # ---------------------------
  allPosIdx <- which(manifesto_data$category == current_class)
  allNegIdx <- which(manifesto_data$category != current_class)
  if (initial_sampling_strategy == "RANDOM") {
    sample_base_population_pos <- allPosIdx
    sample_base_population_neg <- allNegIdx
  } 
  if (initial_sampling_strategy == "BIAS_TIME") {
    idx_bias <- which(as.integer(as.character(manifesto_data[, "year"])) <= 2010)
    sample_base_population_pos <- intersect(allPosIdx, idx_bias)
    sample_base_population_neg <- sample(intersect(allNegIdx, idx_bias))
  }
  if (initial_sampling_strategy == "BIAS_COUNTRY") {
    idx_bias <- which(manifesto_data[, "country"] %in% c("UK", "USA"))
    sample_base_population_pos <- intersect(allPosIdx, idx_bias)
    sample_base_population_neg <- sample(intersect(allNegIdx, idx_bias))
  }
  if (initial_sampling_strategy == "BIAS_PARTY") {
    idx_bias <- which(manifesto_data[, "party"] %in% c("LAB", "LABOUR", "DEM", "GRE"))
    sample_base_population_pos <- intersect(allPosIdx, idx_bias)
    sample_base_population_neg <- sample(intersect(allNegIdx, idx_bias))
  }
  set.seed(1001 * exp_i)
  sample_neg_idx <- sample(sample_base_population_neg, round(initial_training_size / 2))
  sample_pos_idx <- sample(sample_base_population_pos, round(initial_training_size / 2))
  experiment$labels[sample_neg_idx] <- levels(manifesto_data$category)[1]
  experiment$labels[sample_pos_idx] <- levels(manifesto_data$category)[2]
  ##### initial biased sampling end
  
  experiment$active_learning(stop_threshold = 0.99, positive_class = current_class, strategy = "LCB", facets = manifestos, max_iterations = 200, stop_window = 4)
  
  current_exp_result <- data.frame(
    i_sample_strategy = which(initial_sampling_strategies == initial_sampling_strategy),
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
    NegEx = experiment$progress$n_neg,
    Certainty = experiment$progress$stabililty
  )
  
  return(current_exp_result)
}
samplingStrategy_AlResultList_N_Iterations <- do.call(rbind, AlResultList_N_Iterations_List)


# save(samplingStrategy_AlResultList_N_Iterations, file = "5_stopping_w_bias_100iter.RData")
# load("5_stopping_w_bias_100iter.RData")




stop_criterion_matches <- function(v, window = 2, threshold = 0.99) {
  "Stopping criterion for active learning: Stability (see Bloodgood; Vijay-Shanker 2009)"
  b <- v[!is.na(v)] > threshold
  if (length(b) < window) return(0)
  r <- sapply(1:(length(b) - window + 1), FUN = function(x) {
    if (all(b[x:(x + window - 1)])) {
      T
    } else {
      F
    }
  })
  if (!any(r)) {
    return(0)
  }
  return(which(r)[1] + window - 1)
}
all_df <- NULL
for (i_sample_strategy in 1:length(initial_sampling_strategies)) {
  for (activeLearningRun in 1:nALrunsForStatisticalTest) {
    for (codeSet in 1:length(codeSetForClassification)) {
      lidx <- samplingStrategy_AlResultList_N_Iterations$i_sample_strategy == i_sample_strategy
      lidx <- lidx & samplingStrategy_AlResultList_N_Iterations$activeLearningRun == activeLearningRun
      lidx <- lidx & samplingStrategy_AlResultList_N_Iterations$codeSet == codeSet
      filtered_df <- samplingStrategy_AlResultList_N_Iterations[lidx, ]
      stop_row <- stop_criterion_matches(filtered_df$Certainty, window = 3)
      all_df <- rbind(all_df, filtered_df[1:stop_row, ])
    }
  }
}




get_last_points <- function(d) {
  res <- NULL
  for (l in levels(d$codeSet)) {
    d_sel <- d[d$codeSet == l, ]
    res <- rbind(res, d_sel[nrow(d_sel), ])
  }
  return(res)
}

require(ggplot2)
require(reshape2)
require(ggrepel)
category_names <- codeSetForClassification

# FINAL PLOT PAPER
# Plot for the n-th AL run
nAL <- 1 # use specific run for plotting
final_df <- NULL
final_p <- NULL
for (bias_i in 1:length(initial_sampling_strategies)) {
  bias_name <- initial_sampling_strategies[bias_i]
  sel_df <- all_df[all_df[, "i_sample_strategy"] == bias_i, ]
  
  df <- as.data.frame(sel_df[sel_df[, "activeLearningRun"] == nAL, ])
  df <- aggregate(cbind(corCount, corProp, RMSD, PosEx, Certainty) ~ iter + codeSet, df, mean)
  df$codeSet <- as.factor(df$codeSet)
  levels(df$codeSet) <- category_names
  df$iter <- as.factor(df$iter)
  p <- get_last_points(df)
  
  final_df <- rbind(final_df, cbind(strategy = bias_name, df))
  final_p <- rbind(final_p, cbind(strategy = bias_name, p))
}
final_df$strategy <- factor(final_df$strategy, levels = sort(unique(as.character(final_df$strategy)), decreasing = T))
final_p$strategy <- factor(final_p$strategy, levels = sort(unique(as.character(final_p$strategy)), decreasing = T))
ggplot(final_df, aes(x = iter, y = corProp)) + 
  geom_line(aes(group = codeSet, color = codeSet), size = .7) + 
  geom_point(data = final_p, aes(iter, corProp, color = codeSet), show.legend = F, size = 2) + 
  geom_text_repel(data = final_p, aes(label = codeSet), size = 2) +
  facet_wrap(~strategy) + 
  ylab("Pearson's r") + 
  ylim(0, 1) + labs(color = 'Code') +
  scale_x_discrete(breaks = levels(final_df$iter)[c(rep(F, 9), T)], name = "Iteration") +
  scale_color_brewer(palette = "Paired")

ggplot(final_df, aes(x = iter, y = corProp)) + 
  geom_line(aes(group = codeSet, color = codeSet), size = .7) + 
  geom_point(data = final_p, aes(iter, corProp, color = codeSet), show.legend = F, size = 2) + 
  geom_text_repel(data = final_p, aes(label = codeSet), size = 2) +
  facet_wrap(~strategy) + 
  ylab("Pearson's r") + 
  ylim(0, 1) + labs(color = 'Code') +
  scale_x_discrete(breaks = levels(final_df$iter)[c(rep(F, 9), T)], name = "Iteration") +
  scale_color_grey() + theme_bw()


# FINAL TAB PAPER: 100 runs 
# -> MEAN OF corProp with RANDOM initialization at min-stopping iteration of all nActiveLearningRuns
# lower bound!

final_df <- NULL
for (j in 1:length(initial_sampling_strategies)) {
  sel_df <- as.data.frame(samplingStrategy_AlResultList_N_Iterations[samplingStrategy_AlResultList_N_Iterations[, "i_sample_strategy"] == j, ])
  sel_df <- sel_df[!is.na(sel_df$corProp), ]
  sel_df$codeSet <- as.factor(sel_df$codeSet)
  levels(sel_df$codeSet) <- category_names
  mean_corProp <- aggregate(corProp ~ iter + codeSet, sel_df, mean)
  iter_per_code <- aggregate(iter ~ activeLearningRun + codeSet, sel_df, max)
  min_iter <- aggregate(iter ~ codeSet, iter_per_code, min)
  final_df_strat <- NULL
  for (i in 1:length(min_iter$codeSet)) {
    cs <- min_iter$codeSet[i]
    final_df_strat <- rbind(final_df_strat, sel_df[sel_df$codeSet == cs & sel_df$iter == min_iter$iter[i], ])
  }
  final_df <- rbind(final_df, final_df_strat)
}

# USE FOR PAPER: mean corProp
final_df$i_sample_strategy <- as.factor(final_df$i_sample_strategy)
levels(final_df$i_sample_strategy) <- initial_sampling_strategies
ggplot(final_df, aes(x = codeSet, y = corProp, group = codeSet, fill = codeSet)) + 
  geom_boxplot() +
  ylab("Pearson's r") + 
  guides(fill=FALSE) +
  # ylim(0.5, 1) + 
  facet_wrap(~i_sample_strategy) +
  scale_x_discrete(name = "Code") +
  scale_fill_brewer(palette = "Paired")
  # scale_fill_grey() + theme_bw()



# USE FOR PAPER: mean iterations
sel_df <- as.data.frame(all_df)
sel_df <- sel_df[!is.na(sel_df$corProp), ]
sel_df$codeSet <- as.factor(sel_df$codeSet)
levels(sel_df$codeSet) <- category_names
final_df <- NULL
for (cs in levels(sel_df$codeSet)) {
  for (i in 1:nALrunsForStatisticalTest) {
    for (j in 1:length(initial_sampling_strategies)) {
      al_run_df <- sel_df[sel_df$codeSet == cs & sel_df$activeLearningRun == i & sel_df$i_sample_strategy == j, ]
      
      threshold_idx <- nrow(al_run_df)
      threshold_idx <- stop_criterion_matches(al_run_df$Certainty, window = 3)
      
      final_df <- rbind(final_df, al_run_df[threshold_idx, ])
    }
  }
}
# View(final_df)
final_df$i_sample_strategy <- as.factor(final_df$i_sample_strategy)
levels(final_df$i_sample_strategy) <- initial_sampling_strategies
# USE FOR PAPER: mean iterations
ggplot(final_df, aes(x = codeSet, y = iter, group = codeSet, fill = codeSet)) + 
  geom_boxplot() +
  ylab("Iterations") + 
  guides(fill=FALSE) +
  scale_x_discrete(name = "Code") +
  # scale_fill_brewer(palette = "Paired") +
  scale_fill_grey() + theme_bw() +
  facet_wrap(~i_sample_strategy) + 
  coord_flip()




# APPENDIX TABLE A.1
# ------------------
# classifier performance after meeting the stopping criterion in a longevity window of {1,2,3} times in a row

View(head(samplingStrategy_AlResultList_N_Iterations))

final_df <- NULL
sel_df <- as.data.frame(samplingStrategy_AlResultList_N_Iterations)
sel_df <- sel_df[!is.na(sel_df$corProp), ] # throw away NA from early stopping
sel_df <- sel_df[sel_df$i_sample_strategy == 1, ] # only use rnd inital sampling results
for (cs in 1:10) {
  for (i in 1:nALrunsForStatisticalTest) {
    al_run_df <- sel_df[sel_df$codeSet == cs & sel_df$activeLearningRun == i, ]
    # View(al_run_df)
    for (window in 1:4) {
      threshold_idx <- stop_criterion_matches(al_run_df$Certainty, window = window)
      result_when_stopping <- al_run_df[threshold_idx, ]
      result_when_stopping$window <- window
      final_df <- rbind(final_df, result_when_stopping)
    }
  }
}
View(final_df)

# FINAL TAB
selected_cols <- c("iter", "PosEx", "NegEx", "RMSD", "kappa", "corProp")
final_tab <- aggregate(final_df[, selected_cols], by = list(window = final_df$window, codeSet = final_df$codeSet), mean)

improvements <- abs(final_tab$corProp[-nrow(final_tab)] - final_tab$corProp[-1]) * 100 / final_tab$corProp[-nrow(final_tab)]
improvements[rep(c(F,F,F,T), 10)] <- NA
improvements <- c(improvements[40], improvements[1:39])

final_tab[, 1:2] <- final_tab[, 2:1]
colnames(final_tab)[1:2] <- c("codeSet", "window")
extended_tab <- cbind(final_tab, improvements)
extended_tab$iter <- round(extended_tab$iter, digits = 0)
extended_tab$PosEx <- round(extended_tab$PosEx, digits = 0)
extended_tab$NegEx <- round(extended_tab$NegEx, digits = 0)
extended_tab$RMSD <- round(extended_tab$RMSD, digits = 3)
extended_tab$kappa <- round(extended_tab$kappa, digits = 3)
extended_tab$corProp <- round(extended_tab$corProp, digits = 3)
extended_tab$improvements <- paste0(round(extended_tab$improvements, digits = 2), " %")
extended_tab$improvements[extended_tab$improvements == "NA %"] <- "-"
View(extended_tab)
View(extended_tab)