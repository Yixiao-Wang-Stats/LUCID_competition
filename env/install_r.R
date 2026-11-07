# Run from the supplementary root with R_LIBS_USER configured as below.
lib <- Sys.getenv("R_LIBS_USER", unset = "env/R/library")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .Library))
options(repos = c(CRAN = "https://cloud.r-project.org"))
packages <- c("remotes", "Rcpp", "RColorBrewer", "arf", "mlt", "partykit", "trtf", "shapr")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, lib = lib)
# Upstream installation instructions: https://github.com/vittorioorlandi/drbart
if (!requireNamespace("drbart", quietly = TRUE)) {
    remotes::install_github("vittorioorlandi/drbart", ref = "66178ffdbf75c2a0a499cea917f9526073561f98", lib = lib,
                           upgrade = "never", dependencies = NA)
}
required <- c("drbart", "arf", "mlt", "partykit", "trtf", "shapr")
stopifnot(all(vapply(required, requireNamespace, logical(1), quietly = TRUE)))
versions <- vapply(required, function(p) as.character(packageVersion(p)), character(1))
dir.create("results", showWarnings = FALSE)
write.csv(data.frame(package = required, version = versions), "results/r_versions.csv", row.names = FALSE)
cat("R dependencies installed and checked.\n")
