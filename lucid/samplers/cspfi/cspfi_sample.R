suppressPackageStartupMessages({
    library(mlt)
    library(partykit)
    library(trtf)
})

cspfi_sample <- function(fit,X,X_background,n_resamples,seed = 2026) {
    X <- as.data.frame(X)
    X_background <- as.data.frame(X_background)
    feature_size <- ncol(X)
    names(X) <- sprintf("x%03d",seq_len(feature_size) - 1L)
    names(X_background) <- names(X)
    feature_types <- fit$schema$feature_types
    factor_levels <- fit$schema$factor_levels
    target_idx_r <- fit$feature_idx + 1L
    # Apply the saved factor levels to both evaluation observations and background
    # observations.
    for (j in seq_len(feature_size)) {
        if (feature_types[j] %in% c("binary","categorical")) {
            X[[j]] <- factor(X[[j]],levels = factor_levels[[j]])
            X_background[[j]] <- factor(X_background[[j]],levels = factor_levels[[j]])
        } else {
            X[[j]] <- as.numeric(X[[j]])
            X_background[[j]] <- as.numeric(X_background[[j]])
        }
    }
    set.seed(seed)
    if (!is.null(fit$constant_value)) {
        return(matrix(fit$constant_value,nrow = nrow(X),ncol = n_resamples))   # if all values of the target variable are the same then just keep it as constant
    }
    # Assign evaluation and background observations to the same fitted partition.
    if (is.null(fit$model)) {
        X_nodes <- rep(1L,nrow(X))
        reference_nodes <- rep(1L,nrow(X_background))
    } else {
        X_nodes <- predict(fit$model,newdata = X,type = "node")
        reference_nodes <- predict(fit$model,newdata = X_background,type = "node")
    }
    # For each evaluation observation, resample the target feature from background
    # observations in its leaf.
    replacement_values <- t(vapply(seq_len(nrow(X)),function(i) {
        reference_ids <- which(reference_nodes == X_nodes[i])
        replacement_pool <- X_background[[target_idx_r]][reference_ids]
        if (is.factor(replacement_pool)) {
            replacement_pool <- as.numeric(
                as.character(replacement_pool)
            )
        }  # transfer the value into original space for categorical features
        replacement_pool[sample.int(
            length(replacement_pool),
            size = n_resamples,
            replace = TRUE
        )]
    },numeric(n_resamples)))
    replacement_values
}

# Read Python's saved fit and datasets, then return replacements as a CSV matrix.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
    stop("Expected: fit_file X_file X_background_file output_file n_resamples seed.")
}
fit <- readRDS(args[[1L]])
X <- read.csv(args[[2L]],header = FALSE)
X_background <- read.csv(args[[3L]],header = FALSE)
output_file <- args[[4L]]
n_resamples <- as.integer(args[[5L]])
seed <- as.integer(args[[6L]])
replacement_values <- cspfi_sample(
    fit = fit,
    X = X,
    X_background = X_background,
    n_resamples = n_resamples,
    seed = seed
)
write.table(
    replacement_values,
    file = output_file,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE
)