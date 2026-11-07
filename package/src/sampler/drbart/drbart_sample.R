# This is the sample function for drbart
# input is: fit model, x, posterior_id, u
# then we get mu and sigma.
predict_mu_sigma <- function(
    fit,
    mean_trees,
    prec_trees,
    x,
    posterior_id,
    u
) {
    x <- as.matrix(x)
    mean_input <- cbind(u, x)
    mu <- as.numeric(mean_trees$predict_i(t(mean_input), posterior_id - 1L))  # predict mean from i-th posterior
    if (fit$variance == "ux") {
        raw_prec <- as.numeric(prec_trees$predict_prec_i(t(mean_input),posterior_id - 1L))  # predict raw precision from i-th posterior
        sigma <- 1 / sqrt(fit$fit$phistar[posterior_id] * raw_prec)
    } else if (fit$variance == "x") {
        raw_prec <- as.numeric(prec_trees$predict_prec_i(t(x),posterior_id - 1L))
        sigma <- 1 / sqrt(fit$fit$phistar[posterior_id] * raw_prec)
    } else if (fit$variance == "const") {
        sigma <- rep(fit$fit$sigma[posterior_id],nrow(x))
    } else {
        stop("Unsupported DR-BART variance mode.")
    }
    list(mu = mu,sigma = sigma)
}


load_drbart_fit <- function(output_dir) {
    fit_file <- file.path(output_dir, "fit.rds")
    if (!file.exists(fit_file)) {
        stop("fit.rds does not exist.")
    }
    fit <- readRDS(fit_file)
    fit
}

load_drbart_trees <- function(fit) {
    if (!file.exists(fit$mean_file)) {
        stop("DR-BART mean tree file does not exist.")
    }
    if (fit$variance != "const" && !file.exists(fit$prec_file)) {
        stop("DR-BART precision tree file does not exist.")
    }
    TreeSamples <- getFromNamespace("TreeSamples", "drbart") # TreeSamples is a class for handling tree samples in rcpp in drbart
    mean_trees <- TreeSamples$new()
    mean_trees$load(fit$mean_file)
    prec_trees <- NULL
    if (fit$variance != "const") {
        prec_trees <- TreeSamples$new()
        prec_trees$load(fit$prec_file)
    }
    list(mean = mean_trees, prec = prec_trees)
}


truncated_distribution_sampler <- function(fit,mean_trees,prec_trees,x,posterior_id,bounds) {
    # Sample from the DR-BART conditional distribution truncated to the given bounds.
    #
    # In very rare posterior draws, DR-BART can produce Gaussian components with
    # extremely large variance, e.g. N(0.6, 1000000^2). In such cases, naive
    # rejection sampling may fail to obtain a draw within the reasonable support
    # even after many resampling attempts. We therefore use this sampler as a fallback.
    #
    # The idea is to treat each u-bin as a Gaussian mixture component with original
    # weight w_k. For component k, compute
    #
    #   A_k = P(bounds[1] <= Z_k <= bounds[2])
    #       = Phi((bounds[2] - mu_k) / sigma_k)
    #         - Phi((bounds[1] - mu_k) / sigma_k).
    #
    # After truncation, the component weights become
    #
    #   w_k^* proportional to w_k * A_k.
    #
    # Thus, components that place more probability mass inside the allowed support
    # are more likely to be selected. We first sample a component using these
    # adjusted weights, then sample from that component's truncated normal distribution.
    # This truncated distribution is exactly equal
    # p(z | z \in bounds)

    # suppose p(z) = \sum_k w_k f_k(z)
    # p(z | z \in bounds) = p(z)1{z \in buonds} / p(z \in bounds)
    # = \sum_{k} w_k f_k(z) 1{z \in buonds} / \sum_{j} w_j f_j(z)
    # = \sum_{k} (f_k(z) 1{z \in buonds} / A_k) * (w_k A_k / \sum_{j} w_j f_j(z))
    # = \sum_{k} f_k(z | z\in bounds) * w_k^*
    # = \sum_{k} w_k^*  *  f_k(z | z\in bounds) 

    x <- as.matrix(x)  # shape (replacement_sample_size, 1)
    if (length(bounds) != 2L || any(!is.finite(bounds)) || bounds[1L] >= bounds[2L]) {
        stop("Truncated distribution requires finite increasing bounds.")
    }
    cuts <- c(0,fit$fit$ucuts[[posterior_id]],1)
    # for ux mode, we should also save the cuts for precision
    if (fit$variance == "ux") {
        # Include every precision u cut, even if absent from the mean-tree cuts.
        connection <- file(fit$prec_file,"r")
        on.exit(close(connection),add=TRUE)
        header <- scan(connection,nmax=2L,quiet=TRUE)
        precision_cuts <- scan(connection,nmax=header[2L],quiet=TRUE)
        if (length(precision_cuts) != header[2L]) stop("Incomplete precision-tree u cuts.")
        cuts <- c(cuts,precision_cuts)
    }
    cuts <- sort(unique(cuts))
    if (any(!is.finite(cuts)) || any(cuts < 0 | cuts > 1)) stop("Invalid posterior u cuts.")
    widths <- diff(cuts)
    mids <- head(cuts,-1L) + widths/2

    # This is the function returning the log A_k
    interval_log_mass <- function(mu,sigma) {
        a <- (bounds[1L]-mu)/sigma
        b <- (bounds[2L]-mu)/sigma
        # if data is right tail, in order to make numerical stability
        right <- a >= 0
        high <- ifelse(right,pnorm(a,lower.tail=FALSE,log.p=TRUE),pnorm(b,log.p=TRUE))
        low <- ifelse(right,pnorm(b,lower.tail=FALSE,log.p=TRUE),pnorm(a,log.p=TRUE))
        result <- high + log(-expm1(low-high))
        # Resolve CDF cancellation for a density constant to floating precision.
        narrow <- !is.finite(result) & is.finite(a) & is.finite(b) &
            (bounds[2L]-bounds[1L])/sigma * (1+abs(a)) < 1e-8
        result[narrow] <- log(bounds[2L]-bounds[1L])-log(sigma[narrow]) +
            dnorm((bounds[1L]/2+bounds[2L]/2-mu[narrow])/sigma[narrow],log=TRUE)
        result
    }
    
    # sample from f_k(z | z\in bounds)
    # this function is a standard truncated normal sampler. Specific implementation should not be matter here.
    truncated_normal <- function(mu,sigma,lower,upper) {
        a <- (lower-mu)/sigma
        b <- (upper-mu)/sigma
        if ((upper-lower)/sigma * (1+min(abs(a),abs(b))) < 1) {
            peak <- min(max(mu,lower),upper)  
            repeat {
                candidate <- runif(1L,lower,upper)  
                log_ratio <- -0.5*((candidate-peak)/sigma)*((candidate+peak-2*mu)/sigma)
                if (log(runif(1L)) <= log_ratio) return(candidate)
            }
        }
        if (b < 0) return(-truncated_normal(-mu,sigma,-upper,-lower))
        if (a > 0) {
            rate <- a/2 + sqrt((a/2)^2+1)
            repeat {
                offset <- -log1p(-runif(1L)*(-expm1(-rate*((upper-lower)/sigma))))/rate
                z <- a+offset
                if (log(runif(1L)) <= -0.5*(z-rate)^2) {
                    return(min(upper,max(lower,lower+sigma*offset)))
                }
            }
        }
        z <- qnorm(runif(1L,pnorm(a),pnorm(b)))
        min(upper,max(lower,mu+sigma*z))
    }
    replacement <- numeric(nrow(x))
    mixtures <- new.env(parent=emptyenv())
    for (row in seq_len(nrow(x))) {
        key <- paste0("x:",paste(sprintf("%.17g",x[row,]),collapse=","))
        if (!exists(key,mixtures,inherits=FALSE)) {
            parameters <- predict_mu_sigma(fit,mean_trees,prec_trees,
                x[rep(row,length(mids)),,drop=FALSE],posterior_id,mids)
            if (any(!is.finite(parameters$mu)) || any(!is.finite(parameters$sigma)) || any(parameters$sigma <= 0)) {
                stop(sprintf("Invalid truncated Gaussian parameters: posterior_id=%d",posterior_id))
            }
            # log (w_k A_k)
            log_weights <- log(widths)+interval_log_mass(parameters$mu,parameters$sigma)
            if (!any(is.finite(log_weights))) {
                stop(sprintf("No representable bounded mass: posterior_id=%d",posterior_id))
            }
            log_weights[!is.finite(log_weights)] <- -Inf
            parameters$weights <- exp(log_weights-max(log_weights))
            assign(key,parameters,mixtures)
        }
        parameters <- get(key,mixtures,inherits=FALSE)
        # sample a component from w_k*
        component <- sample.int(length(mids),1L,prob=parameters$weights)
        # then since already sampled from w_k* and component is k, we only need to sample from f_k(z | z\in bounds)
        replacement[row] <- truncated_normal(parameters$mu[component],parameters$sigma[component],bounds[1L],bounds[2L])
    }
    replacement
}


# next let's finsih the sample given singular posterior
sample_given_single_posterior <- function(
    fit,
    mean_trees,
    prec_trees,
    x,
    posterior_id,
    replacement_sample_size
) {
    # First, we get the posterior_id
    x <- as.matrix(x)
    # x_repeat <- x[rep(1:nrow(x), each = replacement_sample_size), , drop = FALSE]  # sample x with replacement for replacement_sample_size times
    x_repeat <- x[rep(seq_len(nrow(x)), each = replacement_sample_size), , drop = FALSE]
    replacement <- numeric(nrow(x_repeat))
    pending <- seq_len(nrow(x_repeat))
    # try fit 10000 times 
    for (attempt in seq_len(if (is.null(bounds)) 1L else 10000L)) {
        u <- runif(length(pending))
        parameters <- predict_mu_sigma(fit,mean_trees,prec_trees,x_repeat[pending,,drop=FALSE],posterior_id,u)
        if (any(!is.finite(parameters$mu)) || any(!is.finite(parameters$sigma)) || any(parameters$sigma <= 0)) {
            stop(sprintf("Invalid Gaussian parameters: feature=%s posterior_id=%d",feature_label,posterior_id))
        }
        candidate <- parameters$mu + parameters$sigma * rnorm(length(pending))
        keep <- if (is.null(bounds)) rep(TRUE,length(pending)) else is.finite(candidate) & candidate >= bounds[1L] & candidate <= bounds[2L]
        replacement[pending[keep]] <- candidate[keep]
        pending <- pending[!keep]
        if (!length(pending)) return(replacement)
    }
    message(sprintf("Using truncated_distribution_sampler after 10000 attempts: feature=%s posterior_id=%d remaining=%d bounds=[%.17g,%.17g] fit_dir=%s",feature_label,posterior_id,length(pending),bounds[1L],bounds[2L],fit_dir))
    replacement[pending] <- truncated_distribution_sampler(fit,mean_trees,prec_trees,x_repeat[pending,,drop=FALSE],posterior_id,bounds)
    replacement
}


# Here we implement two ways to sample from the posteriot
# If we only need lucid then we can pick posterior id then pick u epsilon so on
# If we need credible interval we should fix posterior id.
sample_posterior_for_lucid <- function(
    fit,
    mean_trees,
    prec_trees,
    x,
    replacement_sample_size
) {
    posterior_count <- length(fit$fit$ucuts)
    # posterior_ids <- sample.int(
    #     posterior_count,
    #     replacement_sample_size,
    #     replace = TRUE
    # )
    # replacement <- numeric(replacement_sample_size)
    evaluation_sample_size <- nrow(x)
    total_sample_size <- evaluation_sample_size * replacement_sample_size
    posterior_ids <- sample.int(
        posterior_count,
        total_sample_size,
        replace = TRUE
    )
    replacement <- numeric(total_sample_size)
    x_repeat <- x[rep(seq_len(evaluation_sample_size), each = replacement_sample_size), , drop = FALSE]
    for (posterior_id in unique(posterior_ids)) {
        positions <- which(posterior_ids == posterior_id)
        current_replacement <- sample_given_single_posterior(
            fit = fit,
            mean_trees = mean_trees,
            prec_trees = prec_trees,
            x = x_repeat[positions, , drop = FALSE],
            posterior_id = posterior_id,
            replacement_sample_size = 1
        )
        replacement[positions] <- current_replacement
    }
    matrix(
        replacement,
        nrow = evaluation_sample_size,
        ncol = replacement_sample_size,
        byrow = TRUE
    )
}


sample_fixed_posterior_for_credible_interval <- function(
    fit,
    mean_trees,
    prec_trees,
    x,
    posterior_id,
    replacement_sample_size
) {
    evaluation_sample_size <- nrow(x)
    replacement <- sample_given_single_posterior(
        fit = fit,
        mean_trees = mean_trees,
        prec_trees = prec_trees,
        x = x,
        posterior_id = posterior_id,
        replacement_sample_size = replacement_sample_size
    )
    matrix(
        replacement,
        nrow = evaluation_sample_size,
        ncol = replacement_sample_size,
        byrow = TRUE
    )
}



args <- commandArgs(trailingOnly = TRUE)

if (!length(args) %in% c(6L, 7L, 9L, 10L)) {
    stop("Invalid number of arguments.")
}

fit_dir <- args[[1L]]
input_file <- args[[2L]]
output_file <- args[[3L]]
replacement_sample_size <- as.integer(args[[4L]])
mode <- args[[5L]]
seed <- as.integer(args[[6L]])

if (!mode %in% c("lucid", "credible_interval")) {
    stop("mode must be either 'lucid' or 'credible_interval'.")
}

if (replacement_sample_size <= 0) {
    stop("replacement_sample_size must be positive.")
}

base_args <- if (mode == "credible_interval") 7L else 6L
if (!length(args) %in% c(base_args,base_args+3L)) {
    stop("Invalid mode arguments; optional bounds require lower, upper and feature label.")
}
feature_label <- if (length(args) > base_args) args[[base_args+3L]] else basename(fit_dir)
bounds <- if (length(args) > base_args) as.numeric(args[base_args+seq_len(2L)]) else NULL
if (!is.null(bounds) && (any(!is.finite(bounds)) || bounds[1L] >= bounds[2L])) stop("Invalid continuous bounds.")

x <- as.matrix(read.csv(input_file, header = FALSE))
fit <- load_drbart_fit(fit_dir)
trees <- load_drbart_trees(fit)

set.seed(seed)

if (mode == "lucid") {
    replacement <- sample_posterior_for_lucid(
        fit = fit,
        mean_trees = trees$mean,
        prec_trees = trees$prec,
        x = x,
        replacement_sample_size = replacement_sample_size
    )
} else {
    posterior_id <- as.integer(args[[7L]])
    posterior_count <- length(fit$fit$ucuts)
    if (posterior_id < 1L || posterior_id > posterior_count) {
        stop("posterior_id is out of range.")
    }
    replacement <- sample_fixed_posterior_for_credible_interval(
        fit = fit,
        mean_trees = trees$mean,
        prec_trees = trees$prec,
        x = x,
        posterior_id = posterior_id,
        replacement_sample_size = replacement_sample_size
    )
}

write.table(
    replacement,
    file = output_file,
    sep = ",",
    row.names = FALSE,
    col.names = FALSE
)






# # test
# output_dir <- "results/tests/test_package/test_drbart"
# fit <- load_drbart_fit(output_dir)
# trees <- load_drbart_trees(fit)

# print(class(fit))
# print(fit$variance)
# print(is.null(trees$mean))
# print(is.null(trees$prec))

# x <- matrix(c(0.2, -0.5), nrow = 1)
# u <- 0.3
# posterior_id <- 1

# parameters <- predict_mu_sigma(
#     fit = fit,
#     mean_trees = trees$mean,
#     prec_trees = trees$prec,
#     x = x,
#     posterior_id = posterior_id,
#     u = u
# )

# print(parameters$mu)
# print(parameters$sigma)

# set.seed(2026)

# replacement <- sample_given_single_posterior(
#     fit = fit,
#     mean_trees = trees$mean,
#     prec_trees = trees$prec,
#     x = x,
#     posterior_id = 1,
#     replacement_sample_size = 10
# )

# print(replacement)
# print(length(replacement))

# replacement_marginal <- sample_posterior_for_lucid(
#     fit = fit,
#     mean_trees = trees$mean,
#     prec_trees = trees$prec,
#     x = x,
#     replacement_sample_size = 10
# )

# print(replacement_marginal)
# print(length(replacement_marginal))

# replacement_fixed <- sample_fixed_posterior_for_credible_interval(
#     fit = fit,
#     mean_trees = trees$mean,
#     prec_trees = trees$prec,
#     x = x,
#     posterior_id = 1,
#     replacement_sample_size = 10
# )

# print(replacement_fixed)
# print(length(replacement_fixed))
