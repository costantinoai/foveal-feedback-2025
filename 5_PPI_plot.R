#--------------------------------------------------------------------------
# Script for fMRI Data Analysis using Linear Mixed-Effects Models
#
# Purpose:
# This script performs the following tasks:
# 1. Loads and preprocesses fMRI data from a provided CSV file.
# 2. Creates and evaluates linear models for specified predictors (e.g., y_per, y_opp, etc.).
# 3. Computes regression coefficients for interaction terms and generates plots.
# 4. Saves the results and visualization for further analysis and reporting.
#
# Dependencies:
# - data.table: Efficient data manipulation.
# - ggplot2: For creating visualizations.
# - parallel: To enable parallel processing for efficiency.
#--------------------------------------------------------------------------

# Clear the workspace to avoid conflicts with existing objects.
rm(list=ls())

# Load required libraries.
library(data.table) # For data manipulation.
library(ggplot2)    # For creating plots.
library(parallel)   # For parallel computation.

# Set the seed for reproducibility of results.
set.seed(42)

# Configure output options for better readability in the console.
options("width" = 100)

# Load the fMRI dataset from the specified CSV file.
X <- fread('/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2025/results/20250109-212116_perform_PPI/ppi_results.csv') # Ensure the file path is correct.

# Extract and preprocess data for different predictors (e.g., y_per, y_opp).
# Each subset includes response variable (y_fov), predictors, and control variables.
X_per <- X[, .(sub, run, TR, y_fov, y_p, y_per, y_ppi_per, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
X_opp <- X[, .(sub, run, TR, y_fov, y_p, y_opp, y_ppi_opp, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
X_loc <- X[, .(sub, run, TR, y_fov, y_p, y_loc, y_ppi_loc, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
X_ffa <- X[, .(sub, run, TR, y_fov, y_p, y_ffa, y_ppi_ffa, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
#X_a1  <- X[, .(sub, run, TR, y_fov, y_p, y_a1, y_ppi_a1, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]

# Rename columns for consistent processing (y_s for predictor, y_ppi for interaction term).
setnames(X_per, c("y_per", "y_ppi_per"), c("y_s", "y_ppi"))
setnames(X_opp, c("y_opp", "y_ppi_opp"), c("y_s", "y_ppi"))
setnames(X_loc, c("y_loc", "y_ppi_loc"), c("y_s", "y_ppi"))
setnames(X_ffa, c("y_ffa", "y_ppi_ffa"), c("y_s", "y_ppi"))
#setnames(X_a1,  c("y_a1",  "y_ppi_a1"),   c("y_s", "y_ppi"))

# Add a column to indicate the predictor type for each subset.
X_per[, predictor := "per"]
X_opp[, predictor := "opp"]
X_loc[, predictor := "loc"]
X_ffa[, predictor := "ffa"]
#X_a1[, predictor := "a1"]

# Combine all subsets into a single dataset for unified analysis.
#X <- rbindlist(list(X_per, X_opp, X_loc, X_ffa, X_a1))
X <- rbindlist(list(X_per, X_opp, X_loc, X_ffa))

# Initialize an empty list to store model results for each predictor.
d <- list()

# Define the list of predictors to analyze.
predictors <- c("per", "opp", "loc", "ffa")
#predictors <- c("per", "opp", "loc", "ffa", "a1")

# Loop through each predictor and fit a linear model.
for (i in 1:length(predictors)) {
  
  # Print the current predictor for tracking progress.
  print(predictors[i])
  
  # Subset the data for the current predictor.
  XX <- X[predictor == predictors[i], lapply(.SD, mean), .(predictor, TR)]
  
  # Fit a linear model to predict y_fov using the specified predictors and control variables.
  fm <- lm(y_fov ~ y_p + y_s + y_ppi 
           + tx + ty + tz 
           + rx + ry + rz 
           + drift_1 + drift_2 + drift_3 
           + run,
           data = XX
  )
  
  # Store the results (interaction term coefficient and confidence intervals).
  d[[i]] <- data.table(Predictor = predictors[i],
                       Estimate = coef(fm)["y_ppi"],
                       lower = confint(fm)[4, 1],
                       upper = confint(fm)[4, 2])
  
  # Print the summary of the model for inspection.
  print(summary(fm))
}

# Combine results from all predictors into a single data table.
d <- rbindlist(d)

# Create a plot to visualize the regression coefficients for interaction terms.
g <- ggplot(data = d, aes(x = Predictor, y = Estimate)) +
  geom_pointrange(aes(ymin = lower, ymax = upper)) + # Add error bars for confidence intervals.
  geom_hline(yintercept = 0.0, linetype = "dashed") + # Add a reference line at y = 0.
  xlab('') + # Remove x-axis label.
  ylab('PPI regression coefficient') + # Label for y-axis.
  theme_classic() # Use a classic theme for the plot.

# Save the generated plot as a PNG file.
ggsave("/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2025/results/20250109-212116_perform_PPI/ppi_plot.png", g, width = 6, height = 3.5) # Adjust dimensions as needed.

# Print the final data table containing results for all predictors.
print(d)
