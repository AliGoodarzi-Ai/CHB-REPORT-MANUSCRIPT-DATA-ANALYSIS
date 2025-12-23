# ============================================================================
# MIND ELEVATOR: FINAL CORRECT PATH ANALYSIS
# ===============
# 
# RESEARCH QUESTIONS:
# RQ1: How does Mind Elevator influence structural complexity?
# RQ2: How does Mind Elevator influence self-efficacy in argumentation?
# RQ3: How does Mind Elevator influence critical thinking skills?
#
# APPROACH: Path analysis examining what improvements predict final outcomes
#
# VARIABLES:
# Exogenous (Predictors):
#   - TCK: Toulmin Component Knowledge improvement
#   - RED: Critical Thinking (RED framework) improvement
#   - RS: Discussion Relevance ratings
#
# Intermediate:
#   - ASC: Argument Structural Complexity improvement
#
# Endogenous (Outcome):
#   - ASE: Self-Efficacy in Argumentation improvement


library(lavaan)
library(tidyverse)
library(semPlot)

setwd("C:/Users/Ali Goodarzi/Desktop/FINAL")

# ==================
# STEP 1: LOAD DATA


cat("\n")
cat(paste(rep("=", 100), collapse = ""))
cat("\nMIND ELEVATOR PATH ANALYSIS - FINAL CORRECT MODEL\n")
cat("Research Questions: RQ1 (Complexity), RQ2 (Self-Efficacy), RQ3 (Critical Thinking)\n")
cat(paste(rep("=", 100), collapse = ""))
cat("\n")

pre_form <- read.csv("pre_form.csv")
post_form <- read.csv("post_form.csv")
relevance_reddit <- read.csv("relevance reddit.csv")

cat("DATA LOADED:\n")
cat(sprintf("✓ pre_form.csv: %d participants\n", nrow(pre_form)))
cat(sprintf("✓ post_form.csv: %d participants\n", nrow(post_form)))
cat(sprintf("✓ relevance_reddit.csv: %d ratings\n\n", nrow(relevance_reddit)))

# =================================================================
# STEP 2: DEFINE MEASUREMENT SCALES


toulmin_items <- c("Claim", "Grounds", "Warrant", "Backing", "Qualifier", "Rebuttals")
red_items <- c("identify_assumptions", "analyze_evidence", "identify_weaknesses",
               "willing_change_position", "consider_perspectives")

cat("MEASUREMENT INSTRUMENTS:\n")
cat("RQ2 (Self-Efficacy): Toulmin Components\n")
cat(sprintf("  Items: %s\n", paste(toulmin_items, collapse=", ")))
cat("RQ3 (Critical Thinking): RED Framework\n")
cat(sprintf("  Items: %s\n", paste(red_items, collapse=", ")))
cat("RQ1 (Structural Complexity): Argument Coding Scheme (1-4 scale)\n")
cat("RQ2 (Self-Efficacy): Overall Argumentative Ability (1-7 scale)\n")
cat("Moderator: Discussion Relevance (1-7 scale)\n\n")

# ==============
# STEP 3: COMPUTE VARIABLES (CHANGE SCORES ONLY)


cat("COMPUTING CHANGE SCORES (Post - Pre):\n\n")

analysis_df <- data.frame(user_ID = pre_form$user_ID)

# RQ2: Toulmin Component Knowledge (Self-Efficacy in Toulmin)
toulmin_pre <- rowMeans(pre_form[, toulmin_items], na.rm = TRUE)
toulmin_post <- rowMeans(post_form[, toulmin_items], na.rm = TRUE)
analysis_df$TCK <- toulmin_post - toulmin_pre

cat("RQ2 - Toulmin Component Knowledge (Self-Efficacy):\n")
cat(sprintf("  Pre:    M = %.2f, SD = %.2f\n", mean(toulmin_pre), sd(toulmin_pre)))
cat(sprintf("  Post:   M = %.2f, SD = %.2f\n", mean(toulmin_post), sd(toulmin_post)))
cat(sprintf("  Change: M = %.2f, SD = %.2f\n\n", mean(analysis_df$TCK), sd(analysis_df$TCK)))

# RQ3: Critical Thinking (RED Framework)
red_pre <- rowMeans(pre_form[, red_items], na.rm = TRUE)
red_post <- rowMeans(post_form[, red_items], na.rm = TRUE)
analysis_df$RED <- red_post - red_pre

cat("RQ3 - Critical Thinking (RED Framework):\n")
cat(sprintf("  Pre:    M = %.2f, SD = %.2f\n", mean(red_pre), sd(red_pre)))
cat(sprintf("  Post:   M = %.2f, SD = %.2f\n", mean(red_post), sd(red_post)))
cat(sprintf("  Change: M = %.2f, SD = %.2f\n\n", mean(analysis_df$RED), sd(analysis_df$RED)))

# RQ1: Structural Complexity
complexity_pre <- pre_form$structural_complexity
complexity_post <- post_form$structural_complexity
analysis_df$ASC <- complexity_post - complexity_pre

cat("RQ1 - Structural Complexity:\n")
cat(sprintf("  Pre:    M = %.2f, SD = %.2f\n", mean(complexity_pre), sd(complexity_pre)))
cat(sprintf("  Post:   M = %.2f, SD = %.2f\n", mean(complexity_post), sd(complexity_post)))
cat(sprintf("  Change: M = %.2f, SD = %.2f\n\n", mean(analysis_df$ASC), sd(analysis_df$ASC)))

# RQ2: Self-Efficacy in Argumentation
se_pre <- pre_form$self_ability_argument
se_post <- post_form$self_ability_argument
analysis_df$ASE <- se_post - se_pre

cat("RQ2 - Overall Self-Efficacy in Argumentation:\n")
cat(sprintf("  Pre:    M = %.2f, SD = %.2f\n", mean(se_pre), sd(se_pre)))
cat(sprintf("  Post:   M = %.2f, SD = %.2f\n", mean(se_post), sd(se_post)))
cat(sprintf("  Change: M = %.2f, SD = %.2f\n\n", mean(analysis_df$ASE), sd(analysis_df$ASE)))

# Relevance (single measurement during intervention)
analysis_df <- analysis_df %>%
  left_join(relevance_reddit, by = "user_ID") %>%
  rename(RS = relevance_score)

cat("MODERATOR - Discussion Relevance:\n")
cat(sprintf("  M = %.2f, SD = %.2f\n\n", mean(analysis_df$RS), sd(analysis_df$RS)))

# Check data integrity
cat("DATA INTEGRITY:\n")
cat(sprintf("✓ Complete cases: %d / %d\n", nrow(na.omit(analysis_df)), nrow(analysis_df)))
cat(sprintf("✓ Missing values: %d\n\n", sum(is.na(analysis_df))))

# ====
# STEP 4: STANDARDIZE VARIABLES


cat("STANDARDIZATION (z-scores: M=0, SD=1):\n")

analysis_std <- analysis_df %>%
  select(TCK, RED, ASC, ASE, RS) %>%
  mutate(across(everything(), scale)) %>%
  as.data.frame()

for(var in c("TCK", "RED", "ASC", "ASE", "RS")) {
  cat(sprintf("  %-4s: M = %.6f, SD = %.6f\n", 
              var, mean(analysis_std[[var]]), sd(analysis_std[[var]])))
}
cat("\n")

# ===============
# STEP 5: SPECIFY THE MODEL


cat(paste(rep("=", 100), collapse = ""))
cat("\nMODEL SPECIFICATION\n")
cat(paste(rep("=", 100), collapse = ""))
cat("\n")

model_spec <- '
  # 
  # STRUCTURAL PATHS: What predicts Complexity and Self-Efficacy?
  # ========================================================================
  
  # Complexity (RQ1): Can TCK, RED, RS improvements explain complexity?
  ASC ~ b1*TCK + b2*RED + b3*RS
  
  # Self-Efficacy (RQ2): Do TCK, RED, ASC, RS explain self-efficacy?
  ASE ~ b4*TCK + b5*RED + b6*ASC + b7*RS
  
  # ========================================================================
  # COVARIANCES: Correlations between exogenous predictors
  # ========================================================================
  
  # Do TCK and RED changes correlate?
  TCK ~~ RED
  
  # Do TCK changes and relevance correlate?
  TCK ~~ RS
  
  # Do RED changes and relevance correlate?
  RED ~~ RS
  
  # ========================================================================
  # INDIRECT EFFECTS: Do improvements work through complexity?
  # ========================================================================
  
  # Mediation: TCK → ASC → ASE
  indirect_tck := b1 * b6
  
  # Mediation: RED → ASC → ASE
  indirect_red := b2 * b6
  
  # Total effects
  total_tck := b4 + b1*b6
  total_red := b5 + b2*b6
'

cat("EQUATION 1 - Predicting Structural Complexity:\n")
cat("  ASC ~ TCK + RED + RS\n")
cat("  (Does Toulmin improvement, critical thinking improvement, and\n")
cat("   discussion relevance predict complexity improvement?)\n\n")

cat("EQUATION 2 - Predicting Self-Efficacy:\n")
cat("  ASE ~ TCK + RED + ASC + RS\n")
cat("  (Do Toulmin improvement, critical thinking improvement, complexity\n")
cat("   improvement, and discussion relevance predict self-efficacy?)\n\n")

cat("COVARIANCES:\n")
cat("  TCK ~~ RED\n")
cat("  TCK ~~ RS\n")
cat("  RED ~~ RS\n\n")

cat("MEDIATION PATHWAYS:\n")
cat("  Indirect TCK: TCK → ASC → ASE\n")
cat("  Indirect RED: RED → ASC → ASE\n\n")

# ============================================================================
# STEP 6: FIT THE MODEL
# ============================================================================

cat(paste(rep("=", 100), collapse = ""))
cat("\nMODEL ESTIMATION\n")
cat(paste(rep("=", 100), collapse = ""))
cat("\n")

fit <- sem(model_spec,
           data = analysis_std,
           estimator = "ML",
           se = "standard",
           test = "standard")

cat("✓ Model fitted successfully\n")
cat("✓ Estimator: Maximum Likelihood\n")
cat("✓ N = 16 participants\n")
cat("✓ Variables = 5 (TCK, RED, ASC, ASE, RS)\n\n")

summary(fit, fit.measures = TRUE, standardized = TRUE)

# ==========================
# STEP 7: EXTRACT RESULTS
# =====================================================

cat("\n")
cat(paste(rep("=", 100), collapse = ""))
cat("\nFINAL RESULTS\n")
cat(paste(rep("=", 100), collapse = ""))
cat("\n")

params <- parameterestimates(fit)

# Regression paths
cat("REGRESSION PATHS (Standardized Coefficients):\n")
cat(paste(rep("-", 100), collapse = ""))
cat("\n")

paths <- params %>%
  filter(op == "~") %>%
  select(lhs, rhs, est, se, z, pvalue) %>%
  mutate(sig = case_when(
    pvalue < 0.001 ~ "***",
    pvalue < 0.01 ~ "**",
    pvalue < 0.05 ~ "*",
    TRUE ~ "ns"
  )) %>%
  rename("Outcome" = lhs, "Predictor" = rhs, "β" = est, "SE" = se,
         "z-value" = z, "p-value" = pvalue, "Sig." = sig)

print(as.data.frame(paths))
cat("\n")

# Covariances
cat("COVARIANCES (Exogenous Variable Correlations):\n")
cat(paste(rep("-", 100), collapse = ""))
cat("\n")

covars <- params %>%
  filter(op == "~~" & lhs %in% c("TCK", "RED")) %>%
  select(lhs, rhs, est, pvalue) %>%
  mutate(sig = case_when(
    pvalue < 0.001 ~ "***",
    pvalue < 0.01 ~ "**",
    pvalue < 0.05 ~ "*",
    TRUE ~ "ns"
  )) %>%
  rename("Variable 1" = lhs, "Variable 2" = rhs, "Correlation" = est,
         "p-value" = pvalue, "Sig." = sig)

print(as.data.frame(covars))
cat("\n")

# Indirect effects
cat("INDIRECT EFFECTS (Mediation through Complexity):\n")
cat(paste(rep("-", 100), collapse = ""))
cat("\n")

indirect <- params %>%
  filter(op == ":=") %>%
  select(lhs, est, se, pvalue) %>%
  mutate(sig = case_when(
    pvalue < 0.001 ~ "***",
    pvalue < 0.01 ~ "**",
    pvalue < 0.05 ~ "*",
    TRUE ~ "ns"
  )) %>%
  rename("Pathway" = lhs, "Indirect Effect" = est, "SE" = se,
         "p-value" = pvalue, "Sig." = sig)

print(as.data.frame(indirect))
cat("\n")

# Model fit
cat("MODEL FIT INDICES:\n")
cat(paste(rep("-", 100), collapse = ""))
cat("\n")

fit_measures <- fitmeasures(fit, c("chisq", "df", "pvalue", "cfi", "rmsea", "srmr"))
print(fit_measures)
cat("\n")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

write.csv(paths, "FINAL_CORRECT_RegressionPaths.csv", row.names = FALSE)
write.csv(covars, "FINAL_CORRECT_Covariances.csv", row.names = FALSE)
write.csv(indirect, "FINAL_CORRECT_IndirectEffects.csv", row.names = FALSE)

sink("FINAL_CORRECT_FullResults.txt")
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("✓ Results saved:\n")
cat("  - FINAL_CORRECT_RegressionPaths.csv\n")
cat("  - FINAL_CORRECT_Covariances.csv\n")
cat("  - FINAL_CORRECT_IndirectEffects.csv\n")
cat("  - FINAL_CORRECT_FullResults.txt\n\n")

# ============================================================================
# STEP 9: CREATE DIAGRAM
# ============================================================================

png("FINAL_CORRECT_PathDiagram.png", width = 14000, height = 9000, res = 450)

semPaths(fit,
         layout = "tree2",
         rotation = 2,
         what = "std",
         nCharNodes = 0,
         sizeMan = 10,
         sizeLat = 10,
         edge.label.cex = 1.2,
         edge.color = "black",
         edge.label.position = 0.35,
         mar = c(5, 5, 5, 5),
         fade = FALSE,
         residuals = TRUE,
         exoVar = FALSE,
         posCol = c("darkgreen"),
         negCol = c("darkred"),
         weighted = TRUE,
         nDigits = 3,
         style = "ram")

dev.off()

cat("✓ Path diagram: FINAL_CORRECT_PathDiagram.png\n\n")

# ============================================================================
# COMPLETION
# ============================================================================

cat(paste(rep("=", 100), collapse = ""))
cat("\n ANALYSIS COMPLETE - READY FOR THESIS\n")
cat(paste(rep("=", 100), collapse = ""))
cat("\n")

cat("SUMMARY:\n")
cat("- Variables: 5 (Exogenous: TCK, RED, RS | Mediator: ASC | Outcome: ASE)\n")
cat("- Sample: N = 16 participants\n")
cat("- Design: Saturated path analysis (df = 0, CFI = 1.0)\n")
cat("- RQs answered: All three (Complexity, Self-Efficacy, Critical Thinking)\n")
cat("- Output: One correct model, one set of results, one diagram\n\n")

