library(EcoacousticUtilities)
df <- read.csv("burnt_only_observations.csv", stringsAsFactors = FALSE)
ids <- df$occurrenceID[nchar(df$occurrenceID) > 0 & !is.na(df$occurrenceID)]
cat("Starting download of", length(ids), "observations at", format(Sys.time()), "\n")
n <- get_inat_images(
  observation_ids = ids,
  image_size      = "original",
  out_dir         = "inat_burnt_images"
)
cat("Requested IDs:", n, "at", format(Sys.time()), "\n")
