# 01 -- Acquire LEDGAR (LexGLUE) and record provenance.
#
# Output: data/raw/*.parquet (git-ignored), results/01_provenance.csv,
#         results/01_label_map.csv

source("analysis/_setup.R")

ledgar_download(paths$raw)
ledgar <- ledgar_read(paths$raw)

provenance <- tibble(split = names(ledgar_urls()), url = ledgar_urls()) |>
  mutate(
    file = file.path(paths$raw, paste0(split, ".parquet")),
    bytes = file.size(file),
    sha256 = map_chr(file, \(f) system2("sha256sum", f, stdout = TRUE) |> word(1)),
    rows = map_int(split, \(s) sum(ledgar$split == s))
  )
write_result(provenance, "01_provenance")

label_map <- ledgar |> distinct(label_id, label) |> arrange(label_id)
write_result(label_map, "01_label_map")

# Structural checks: fail loudly if the download is not what we expect.
stopifnot(
  nrow(ledgar) == 80000,
  identical(as.integer(table(ledgar$split)), c(60000L, 10000L, 10000L)),
  nlevels(ledgar$label) == 100,
  !anyNA(ledgar$text), !anyNA(ledgar$label),
  n_distinct(ledgar$id) == nrow(ledgar)
)
glimpse(ledgar)
