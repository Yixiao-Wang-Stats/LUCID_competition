# This is the sample function for ARF.
# input is:
#     fitted psi
#     x
#     feature_idx
#     n_resamples
#
# output:
#     replacement samples


load_arf_psi <- function(output_dir) {
    psi_file <- file.path(output_dir,"psi.rds")
    if (!file.exists(psi_file)) {
        stop("psi.rds does not exist.")
    }
    psi <- readRDS(psi_file)
    psi
}


sample_arf <- function(
    psi,
    x,
    feature_idx,
    n_resamples,
    schema
) {
    x <- as.matrix(x)
    n_eval <- nrow(x)
    feature_size <- ncol(x)

    if (feature_idx < 0L || feature_idx >= feature_size) {
        stop("feature_idx is out of range.")
    }

    x <- as.data.frame(x)
    names(x) <- sprintf("x%03d",seq_len(ncol(x)) - 1L)

    if (ncol(x) != length(schema$feature_types)) {
        stop("x has the wrong number of features.")
    }

    target_idx_r <- feature_idx + 1L  # the target feature index in 1 based
    # Recreate training factor levels and reject unseen categories in conditioning features.
    for (j in seq_len(ncol(x))) {
        if (schema$feature_types[j] %in% c("binary","categorical")) {
            values <- as.character(x[[j]])
            training_levels <- schema$factor_levels[[j]]
            if (j != target_idx_r &&
                any(!is.na(values) & !(values %in% training_levels))) {
                stop("Conditioning data contain unseen category levels.")
            }
            x[[j]] <- factor(values,levels = training_levels)
        } else {
            x[[j]] <- as.numeric(x[[j]])
        }
    }
    
    conditioning_indices <- setdiff(seq_len(feature_size),target_idx_r)  # remaining feature indices

    # Condition on all features except the target feature; each evidence row is an
    # evaluation observation.
    evidence <- x[
        ,
        conditioning_indices,
        drop = FALSE
    ]  # x_{-j}
    # generate replacement samples
    synthetic <- arf::forge(
        params = psi,
        n_synth = n_resamples,
        evidence = evidence,
        evidence_row_mode = "separate"
    )

    target_name <- names(x)[target_idx_r]
    replacement <- synthetic[[target_name]]
    if (is.factor(replacement) || is.character(replacement)) {
        replacement <- as.numeric(as.character(replacement))
    }

    # Return one row per evaluation observation, with its replacement draws in columns.
    matrix(
        replacement,
        nrow = n_eval,
        ncol = n_resamples,
        byrow = TRUE
    )
}


# Load Python's sampling request, generate replacements, and write a header-free CSV
# response.
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6L) {
    stop("Invalid number of arguments.")
}

fit_dir <- args[[1L]]
input_file <- args[[2L]]
output_file <- args[[3L]]
feature_idx <- as.integer(args[[4L]])
n_resamples <- as.integer(args[[5L]])
seed <- as.integer(args[[6L]])

if (n_resamples <= 0) {
    stop("n_resamples must be positive.")
}

x <- as.matrix(read.csv(input_file,header = FALSE))
psi <- load_arf_psi(fit_dir)

set.seed(seed)

schema <- readRDS(file.path(fit_dir,"schema.rds"))
replacement <- sample_arf(
    psi = psi,
    x = x,
    feature_idx = feature_idx,
    n_resamples = n_resamples,
    schema = schema
)

write.table(
    replacement,
    file = output_file,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE
)