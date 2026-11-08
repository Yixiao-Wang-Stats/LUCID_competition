suppressPackageStartupMessages({
    library(mlt)
    library(partykit)
    library(trtf)
})

cspfi_sample <- function(fit,X,X_reference,n_resamples,seed = 2026) {
    X <- as.data.frame(X)
    X_reference <- as.data.frame(X_reference)
    feature_size <- ncol(X)
    names(X) <- sprintf("x%03d",seq_len(feature_size) - 1L)
    names(X_reference) <- names(X)
    feature_types <- fit$schema$feature_types
    factor_levels <- fit$schema$factor_levels
    target_idx_r <- fit$feature_idx + 1L
    for (j in seq_len(feature_size)) {
        if (feature_types[j] %in% c("binary","categorical")) {
            X[[j]] <- factor(X[[j]],levels = factor_levels[[j]])
            X_reference[[j]] <- factor(X_reference[[j]],levels = factor_levels[[j]])
        } else {
            X[[j]] <- as.numeric(X[[j]])
            X_reference[[j]] <- as.numeric(X_reference[[j]])
        }
    }
    set.seed(seed)
    if (!is.null(fit$constant_value)) {
        return(matrix(fit$constant_value,nrow = nrow(X),ncol = n_resamples))   # if all values of the target variable are the same then just keep it as constant
    }
    if (is.null(fit$model)) {
        X_nodes <- rep(1L,nrow(X))
        reference_nodes <- rep(1L,nrow(X_reference))
    } else {
        X_nodes <- predict(fit$model,newdata = X,type = "node")
        reference_nodes <- predict(fit$model,newdata = X_reference,type = "node")
    }
    replacement_values <- t(vapply(seq_len(nrow(X)),function(i) {
        reference_ids <- which(reference_nodes == X_nodes[i])
        replacement_pool <- X_reference[[target_idx_r]][reference_ids]
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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
    stop("Expected: fit_file X_file X_reference_file output_file n_resamples seed.")
}
fit <- readRDS(args[[1L]])
X <- read.csv(args[[2L]],header = FALSE)
X_reference <- read.csv(args[[3L]],header = FALSE)
output_file <- args[[4L]]
n_resamples <- as.integer(args[[5L]])
seed <- as.integer(args[[6L]])
replacement_values <- cspfi_sample(
    fit = fit,
    X = X,
    X_reference = X_reference,
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