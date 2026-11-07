suppressPackageStartupMessages({
    library(mlt)
    library(partykit)
    library(trtf)
})

cspfi_fit <- function(X_train,output_dir,feature_types,feature_idx,seed = 2026) {
    X_train <- as.data.frame(X_train)
    feature_size <- ncol(X_train)
    names(X_train) <- sprintf("x%03d",seq_len(feature_size) - 1L)
    if (length(feature_types) != feature_size) {
        stop("feature_types must have one entry per feature.")
    }
    if (feature_idx < 0L || feature_idx >= feature_size) {
        stop("feature_idx is out of range.")
    }
    if (anyNA(X_train)) {
        stop("X_train must not contain missing values.")
    }
    for (j in seq_len(feature_size)) {
        if (feature_types[j] %in% c("binary","categorical")) {
            if (feature_types[j] == "binary" && length(unique(X_train[[j]])) > 2L) {
                stop("A binary feature has more than two observed values.")
            }
            X_train[[j]] <- factor(X_train[[j]])
        } else if (feature_types[j] == "continuous") {
            X_train[[j]] <- as.numeric(X_train[[j]])
        } else {
            stop("Unknown feature type.")
        }
    }
    # above are just some check
    target_idx_r <- feature_idx + 1L
    target_name <- names(X_train)[target_idx_r]  # feature j
    y <- X_train[[target_idx_r]]
    schema <- list(
        feature_types = feature_types,
        factor_levels = lapply(X_train,function(column) {
            if (is.factor(column)) levels(column) else NULL
        })
    )
    set.seed(seed)
    model <- NULL
    constant_value <- NULL
    # below follow the same logic of source code in cspfi
    if (length(unique(y)) == 1L) {
        constant_value <- as.numeric(as.character(y[1L]))
    } else if (feature_size > 1L) {
        control <- partykit::ctree_control(minbucket = 30L)
        if (feature_types[target_idx_r] == "continuous") {
            yvar <- numeric_var(target_name,support = range(y))
            basis <- Bernstein_basis(yvar,order = 5L,ui = "incr")
            transformation_model <- ctm(response = basis,todistr = "Normal",data = X_train)
            formula <- as.formula(sprintf("%s ~ 1 | .",target_name))
            model <- trtf::trafotree(
                transformation_model,
                formula = formula,
                data = X_train,
                control = control
            )
        } else {
            formula <- as.formula(sprintf("%s ~ .",target_name))
            model <- partykit::ctree(
                formula,
                data = X_train,
                control = control
            )
        }
    }
    fit <- list(
        model = model,
        feature_idx = feature_idx,
        target_name = target_name,
        schema = schema,
        constant_value = constant_value
    )
    dir.create(output_dir,showWarnings = FALSE,recursive = TRUE)
    saveRDS(fit,file = file.path(output_dir,"fit.rds"))
    invisible(fit)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
    stop("Expected: X_train_file output_dir seed feature_types_file feature_idx.")
}
X_train <- read.csv(args[[1L]],header = FALSE)
output_dir <- args[[2L]]
seed <- as.integer(args[[3L]])
feature_types <- readLines(args[[4L]],warn = FALSE)
feature_idx <- as.integer(args[[5L]])
cspfi_fit(
    X_train = X_train,
    output_dir = output_dir,
    feature_types = feature_types,
    feature_idx = feature_idx,
    seed = seed
)