# input:
#     X_train
#     y_train
#     output_dir
#     seed
#     nburn
#     nsim
#     nthin
#     m_mean
#     m_var
#     variance

# fit:
#     drbart::drbart(...)

# save:
#     fit.rds
#     dr_bart_mean.txt
#     dr_bart_prec.txt

# print(formals(drbart::drbart))
# For our code we only control main parameters:nburn,nsim, nthin, m_mean,m_var,variance and save mean_file and prec_file


drbart_fit <- function(
    X_train,
    y_train,
    output_dir,
    seed = 2026,
    nburn = 5000,
    nsim = 5000,
    nthin = 1,
    m_mean = 200,
    m_var = 100,
    variance = "ux"
) {
    # Validate conditioning predictors and the continuous target before running MCMC.
    X_train <- as.matrix(X_train)
    y_train <- as.vector(y_train)
    if (nrow(X_train) != length(y_train)) {
        stop("X_train and y_train must have the same number of observations.")
    }
    dir.create(output_dir, showWarnings = FALSE)
    mean_file <- file.path(output_dir, "dr_bart_mean.txt")
    prec_file <- file.path(output_dir, "dr_bart_prec.txt")
    fit_file <- file.path(output_dir, "fit.rds")

    # Fit the conditional density and save mean and precision tree histories to the supplied
    # paths.
    set.seed(seed)
    fit <- drbart::drbart(
        y = y_train,
        x = X_train,
        nburn = nburn,
        nsim = nsim,
        nthin = nthin,
        m_mean = m_mean,
        m_var = m_var,
        variance = variance,
        mean_file = mean_file,
        prec_file = prec_file
    )
    saveRDS(fit, file=fit_file)
    # Record the actual saved posterior count so Python can select valid draw IDs.
    posterior_count <- length(fit$fit$ucuts) 
    writeLines(as.character(posterior_count),file.path(output_dir, "posterior_count.txt")) 
    invisible(fit)
}

# Translate Python's positional arguments into fitting data and MCMC settings.
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 10L) {
    stop("Invalid number of arguments.")
}

X_train_file <- args[[1L]]
y_train_file <- args[[2L]]
output_dir <- args[[3L]]
seed <- as.integer(args[[4L]])
nburn <- as.integer(args[[5L]])
nsim <- as.integer(args[[6L]])
nthin <- as.integer(args[[7L]])
m_mean <- as.integer(args[[8L]])
m_var <- as.integer(args[[9L]])
variance <- args[[10L]]

X_train <- as.matrix(read.csv(X_train_file, header = FALSE))
y_train <- as.numeric(read.csv(y_train_file, header = FALSE)[, 1L])

drbart_fit(
    X_train = X_train,
    y_train = y_train,
    output_dir = output_dir,
    seed = seed,
    nburn = nburn,
    nsim = nsim,
    nthin = nthin,
    m_mean = m_mean,
    m_var = m_var,
    variance = variance
)








# # test
# set.seed(2026)

# n <- 100

# X_train <- cbind(
#     x1 = rnorm(n),
#     x2 = rnorm(n)
# )

# y_train <- (
#     2 * X_train[, 1]
#     - X_train[, 2]
#     + rnorm(n)
# )
# fit <- drbart_fit(
#     X_train = X_train,
#     y_train = y_train,
#     output_dir = "results/tests/test_package/test_drbart",
#     seed = 2026,
#     nburn = 10,
#     nsim = 20,
#     nthin = 1,
#     m_mean = 20,
#     m_var = 10,
#     variance = "ux"
# )
