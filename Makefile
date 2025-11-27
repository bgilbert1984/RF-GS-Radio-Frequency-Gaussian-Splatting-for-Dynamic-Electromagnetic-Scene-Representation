# Makefile for RF-GS CVPR 2026 Paper
# Usage: make pdf

PAPER = RF_GS_CVPR2026_Paper
FIGURES_DIR = figures
PYTHON_ENV = /home/bgilbert/rf_quantum_env/bin/python

# Default target
pdf: $(PAPER).pdf

# Generate figures if they don't exist
figures: $(FIGURES_DIR)/teaser.pdf $(FIGURES_DIR)/qualitative.pdf $(FIGURES_DIR)/realworld_deployment.pdf $(FIGURES_DIR)/temporal_analysis.pdf

$(FIGURES_DIR)/teaser.pdf $(FIGURES_DIR)/qualitative.pdf $(FIGURES_DIR)/realworld_deployment.pdf $(FIGURES_DIR)/temporal_analysis.pdf: generate_paper_figures.py
	@echo "Generating paper figures..."
	$(PYTHON_ENV) generate_paper_figures.py

# Compile PDF
$(PAPER).pdf: $(PAPER).tex references.bib figures
	@echo "Compiling LaTeX..."
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex
	@echo "PDF compilation complete: $(PAPER).pdf"

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz

# Clean everything including PDF and figures
clean-all: clean
	rm -f $(PAPER).pdf
	rm -rf $(FIGURES_DIR)/*

# View PDF
view: $(PAPER).pdf
	@echo "Opening PDF..."
	xdg-open $(PAPER).pdf 2>/dev/null || open $(PAPER).pdf 2>/dev/null || echo "Please open $(PAPER).pdf manually"

# Check LaTeX syntax
check:
	@echo "Checking LaTeX syntax..."
	lacheck $(PAPER).tex

# Word count (approximate)
wordcount: $(PAPER).tex
	@echo "Approximate word count:"
	detex $(PAPER).tex | wc -w

# Submit to arXiv (prepare submission package)
arxiv: $(PAPER).pdf
	@echo "Preparing arXiv submission..."
	mkdir -p arxiv_submission
	cp $(PAPER).tex references.bib arxiv_submission/
	cp -r $(FIGURES_DIR) arxiv_submission/
	cd arxiv_submission && tar -czf ../$(PAPER)_arxiv.tar.gz *
	@echo "ArXiv package ready: $(PAPER)_arxiv.tar.gz"

# Help
help:
	@echo "Available targets:"
	@echo "  pdf       - Compile the paper (default)"
	@echo "  figures   - Generate all figures"
	@echo "  clean     - Remove auxiliary files"
	@echo "  clean-all - Remove all generated files"
	@echo "  view      - Open the compiled PDF"
	@echo "  check     - Check LaTeX syntax"
	@echo "  wordcount - Count words (approximate)"
	@echo "  arxiv     - Prepare arXiv submission package"

.PHONY: pdf figures clean clean-all view check wordcount arxiv help