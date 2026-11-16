# Reproduce the LEDGAR study end to end. Run from this directory.
#
#   make setup     install R dependencies and the provisionr package
#   make all       run every analysis step in order, then render the report
#   make check     unit tests + R CMD check for provisionr
#   make board     export the epiq board to docs/epiq/
#
# Individual steps: `make 07` runs analysis/07_*.R, and so on. Expensive
# steps cache their intermediate results under data/processed/.

RSCRIPT := Rscript
STEPS   := 01 02 03 04 05 06 07 08 09 10 11 12 13

.PHONY: all setup check report board clean $(STEPS)

all: $(STEPS) report

setup:
	$(RSCRIPT) tools/install_deps.R
	R CMD INSTALL provisionr

$(STEPS):
	$(RSCRIPT) $(wildcard analysis/$@_*.R)

report:
	$(RSCRIPT) -e 'rmarkdown::render("reports/report.Rmd", quiet = TRUE)'

check:
	$(RSCRIPT) -e 'devtools::test("provisionr")'
	R CMD build provisionr && _R_CHECK_FORCE_SUGGESTS_=false R CMD check --no-manual provisionr_*.tar.gz

board:
	python3 tools/epiq.py export docs/epiq

clean:
	rm -rf data/processed models provisionr.Rcheck provisionr_*.tar.gz
