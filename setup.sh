#!/bin/bash

# Setup script for RF-GS CVPR 2026 Paper
# This script checks for required dependencies and sets up the environment

echo "=== RF-GS Paper Setup ==="

# Check for LaTeX installation
if ! command -v pdflatex &> /dev/null; then
    echo "❌ LaTeX not found. Please install LaTeX:"
    echo "   Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended"
    echo "   macOS: brew install --cask mactex"
    echo "   Windows: Install MiKTeX or TeX Live"
    exit 1
else
    echo "✅ LaTeX found: $(pdflatex --version | head -n1)"
fi

# Check for BibTeX
if ! command -v bibtex &> /dev/null; then
    echo "❌ BibTeX not found. Usually comes with LaTeX installation."
    exit 1
else
    echo "✅ BibTeX found"
fi

# Check Python environment
if [ -f "/home/bgilbert/rf_quantum_env/bin/python" ]; then
    echo "✅ Python virtual environment found"
else
    echo "❌ Python virtual environment not found at expected location"
    echo "   Please ensure the environment is properly configured"
    exit 1
fi

# Generate figures
echo "🎨 Generating paper figures..."
if /home/bgilbert/rf_quantum_env/bin/python generate_paper_figures.py; then
    echo "✅ All figures generated successfully"
else
    echo "❌ Figure generation failed"
    exit 1
fi

# Test LaTeX compilation
echo "📝 Testing LaTeX compilation..."
if make pdf; then
    echo "✅ Paper compiled successfully!"
    echo ""
    echo "🎉 Setup complete! Your paper is ready for submission."
    echo ""
    echo "Next steps:"
    echo "1. Review the compiled PDF: RF_GS_CVPR2026_Paper.pdf"
    echo "2. Add your experimental results to replace placeholders"
    echo "3. Update author information"
    echo "4. Generate real figures from your RF-GS implementation"
    echo ""
    echo "Available commands:"
    echo "  make pdf      - Recompile the paper"
    echo "  make view     - Open the PDF"
    echo "  make clean    - Clean temporary files"
    echo "  make arxiv    - Prepare arXiv submission"
else
    echo "❌ LaTeX compilation failed. Check error messages above."
    exit 1
fi