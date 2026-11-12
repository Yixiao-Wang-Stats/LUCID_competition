arf_fit <- function(X_train,output_dir,feature_types,seed = 2026) {
    X_train <- as.data.frame(X_train)
    names(X_train) <- sprintf("x%03d",seq_len(ncol(X_train)) - 1L)
    if (length(feature_types) != ncol(X_train)) {
        stop("feature_types must have one entry per feature.")
    }
    # Restore categorical columns as factors before estimating the joint feature
    # distribution.
    for (j in seq_len(ncol(X_train))) {
        if (feature_types[j] %in% c("binary","categorical")) {
            if (feature_types[j] == "binary" &&
                length(unique(na.omit(X_train[[j]]))) > 2L) {
                stop("A binary feature has more than two observed values.")
            }
            X_train[[j]] <- factor(X_train[[j]])
        } else if (feature_types[j] == "continuous") {
            X_train[[j]] <- as.numeric(X_train[[j]])
        } else {
            stop("Unknown feature type.")
        }
    }
    dir.create(output_dir,showWarnings = FALSE,recursive = TRUE)
    set.seed(seed)
    
    # Train the ARF
    fit <- arf::adversarial_rf(x = X_train)
    
    # Estimate distribution parameters
    psi <- arf::forde(arf = fit,x = X_train)
    # Save factor levels so sampling can reconstruct the same feature schema.
    schema <- list(
        feature_types = feature_types,
        factor_levels = lapply(X_train,function(column) {
            if (is.factor(column)) levels(column) else NULL
        })
    )
    # Persist the forest, leaf-density parameters, and schema for Python to reuse.
    saveRDS(fit,file = file.path(output_dir,"fit.rds"))
    saveRDS(psi,file = file.path(output_dir,"psi.rds"))
    saveRDS(schema,file = file.path(output_dir,"schema.rds"))
    invisible(fit)
}

# Read the data paths and seed passed by the Python fitting wrapper.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
    stop("Invalid number of arguments.")
}
X_train_file <- args[[1L]]
output_dir <- args[[2L]]
seed <- as.integer(args[[3L]])
feature_types <- readLines(args[[4L]],warn = FALSE)
X_train <- read.csv(X_train_file,header = FALSE)
arf_fit(
    X_train = X_train,
    output_dir = output_dir,
    feature_types = feature_types,
    seed = seed
)


