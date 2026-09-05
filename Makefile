.PHONY: all pass30 pass31 tables paper checksums clean

all: pass30 pass31 tables

pass30:
	python scripts/run_role_stability_pass30.py

pass31:
	python scripts/run_role_stability_pass31.py

tables:
	python scripts/make_tables_figures_pass31.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

checksums:
	python scripts/validate_checksums.py

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
