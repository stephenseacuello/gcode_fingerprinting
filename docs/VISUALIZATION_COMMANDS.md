# High-Quality Visualization Commands

## ✅ What Was Created

**New Script:** `scripts/regenerate_visuals.py`
- Regenerates high-quality visualizations from existing evaluation results
- No model inference needed (fast!)
- Multiple quality presets (publication, presentation, poster, web)
- Multi-format export (PNG @ 300 DPI, SVG, PDF)
- Professional styling with configurable fonts and colors

**Test Run Completed:** ✓
```
reports/publication_quality/
├── bar_charts/
│   ├── accuracy_comparison.png (300 DPI, 3.0 MB)
│   ├── accuracy_comparison.svg (46 KB, scalable)
│   └── accuracy_comparison.pdf (23 KB, print-ready)
```

---

## 📊 COPY-PASTE COMMANDS

### **1. Publication Quality (Papers/Journals)**

High DPI, vector graphics, serif fonts, colorblind-friendly palette

```bash
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/publication \
    --preset publication \
    --dpi 300
```

**Outputs:**
- PNG @ 300 DPI (high resolution for print)
- SVG (infinitely scalable, editable in Inkscape/Illustrator)
- PDF (ready for LaTeX/Word documents)

**Best for:**
- Journal submissions
- Conference papers
- Technical reports
- Thesis/dissertation

---

### **2. Presentation Quality (Slides/Talks)**

Large fonts, bright colors, high contrast

```bash
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/presentation \
    --preset presentation \
    --dpi 200
```

**Outputs:**
- PNG @ 200 DPI (optimized for slides)

**Best for:**
- PowerPoint/Keynote presentations
- Conference talks
- Video recordings
- Zoom presentations

---

### **3. Poster Quality (Conference Posters)**

Extra large fonts, very high DPI, bold colors

```bash
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/poster \
    --preset poster \
    --dpi 300
```

**Outputs:**
- PNG @ 300 DPI
- PDF (vector, print-ready)

**Best for:**
- Academic posters
- Large format printing
- Conference displays

---

### **4. Web Quality (Websites/Dashboards)**

Moderate DPI, optimized file sizes, sans-serif fonts

```bash
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/web \
    --preset web \
    --dpi 150
```

**Outputs:**
- PNG @ 150 DPI (smaller files, fast loading)

**Best for:**
- GitHub README
- Project websites
- Online documentation
- Blog posts

---

### **5. Custom Settings**

Override any preset with custom parameters:

```bash
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/custom \
    --preset publication \
    --dpi 600 \
    --font-scale 2.0 \
    --formats png svg pdf
```

**Parameters:**
- `--dpi`: Resolution (72-600)
- `--font-scale`: Font size multiplier (0.5-3.0)
- `--formats`: Output formats (png, svg, pdf)

---

## 🎨 QUALITY COMPARISON

| Preset | DPI | Font Scale | Figure Size | Formats | Use Case |
|--------|-----|------------|-------------|---------|----------|
| publication | 300 | 1.3x | 1.2x | PNG, SVG, PDF | Papers, journals |
| presentation | 200 | 1.8x | 1.4x | PNG | Slides, talks |
| poster | 300 | 2.2x | 1.6x | PNG, PDF | Posters, printing |
| web | 150 | 1.2x | 1.0x | PNG | Websites, blogs |

---

## 📁 VIEW GENERATED VISUALS

### **Open Specific Files:**

```bash
# Publication quality
open reports/publication/bar_charts/accuracy_comparison.png
open reports/publication/bar_charts/accuracy_comparison.pdf

# Presentation quality
open reports/presentation/bar_charts/accuracy_comparison.png

# All visuals
open reports/publication/
```

### **Compare Formats:**

```bash
# View PNG (raster, fixed resolution)
open reports/publication/bar_charts/accuracy_comparison.png

# View SVG (vector, scalable, editable)
open reports/publication/bar_charts/accuracy_comparison.svg

# View PDF (vector, print-ready)
open reports/publication/bar_charts/accuracy_comparison.pdf
```

---

## 🔄 REGENERATE FROM DIFFERENT EVALUATION RUNS

The script works with ANY comprehensive evaluation results:

```bash
# From focal loss model evaluation
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_focal_loss \
    --output reports/focal_loss_publication \
    --preset publication

# From comprehensive sweep evaluation
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/sweep_publication \
    --preset publication

# From future models (after training with fixed code)
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_FIXED_comprehensive \
    --output reports/FIXED_publication \
    --preset publication
```

---

## 📊 BATCH GENERATE ALL PRESETS

Create all quality versions at once:

```bash
# Generate publication quality
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/publication \
    --preset publication

# Generate presentation quality
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/presentation \
    --preset presentation

# Generate poster quality
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/poster \
    --preset poster

# Generate web quality
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/web \
    --preset web

echo "✅ Generated all quality presets!"
echo "View results:"
echo "  Publication: reports/publication/"
echo "  Presentation: reports/presentation/"
echo "  Poster: reports/poster/"
echo "  Web: reports/web/"
```

---

## 🎯 RECOMMENDED WORKFLOW

### **For Academic Paper:**

1. Train new model with fixed code
2. Run comprehensive evaluation
3. Generate publication quality visuals
4. Use PDF/SVG in LaTeX document

```bash
# After model training completes...
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/comprehensive_evaluation.py \
    --checkpoint outputs/FIXED_composite/checkpoint_best.pt \
    --test-data outputs/processed_hybrid/test_sequences.npz \
    --output reports/paper_evaluation

# Regenerate with publication quality
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/paper_evaluation \
    --output reports/paper_figures \
    --preset publication \
    --dpi 300

# Use the PDFs in your LaTeX paper
# figures are in: reports/paper_figures/bar_charts/*.pdf
```

### **For Presentation:**

```bash
# Use existing evaluation results
.venv/bin/python scripts/regenerate_visuals.py \
    --input reports/test_comprehensive \
    --output reports/presentation_slides \
    --preset presentation

# Insert PNGs into PowerPoint/Keynote
# figures are in: reports/presentation_slides/bar_charts/*.png
```

---

## 🆚 BEFORE vs AFTER COMPARISON

### **Before (Default Quality):**
- DPI: 100
- File size: 250 KB
- Format: PNG only
- Fonts: Small, hard to read
- Colors: Basic matplotlib

### **After (Publication Quality):**
- DPI: 300 (3x higher resolution)
- File size: 3.0 MB PNG, 46 KB SVG, 23 KB PDF
- Formats: PNG + SVG + PDF
- Fonts: Large, bold, professional
- Colors: Colorblind-friendly palette

**Quality Improvement:** ~3x resolution, professional styling, multiple formats

---

## 💡 TIPS

**For Best Results:**

1. **Use SVG for presentations** - infinitely scalable, looks sharp at any size
2. **Use PDF for papers** - vector format, accepted by journals
3. **Use high-DPI PNG for posters** - needed for large format printing
4. **Use low-DPI PNG for web** - faster loading, smaller files

**File Size Guide:**

- Web PNG (150 DPI): ~500 KB - 1 MB
- Presentation PNG (200 DPI): ~1-2 MB
- Publication PNG (300 DPI): ~2-4 MB
- SVG (vector): ~20-100 KB (tiny!)
- PDF (vector): ~20-50 KB (tiny!)

**Recommendation:** Use vector formats (SVG/PDF) whenever possible for smallest files and perfect scaling!

---

## 📝 NEXT STEPS

1. ✅ **View existing publication quality visuals:**
   ```bash
   open reports/publication_quality/
   ```

2. ✅ **Generate all presets for current evaluation:**
   ```bash
   # Run the batch command above
   ```

3. ⏳ **After training new model with fixed code:**
   ```bash
   # Run comprehensive evaluation
   # Then regenerate visuals with publication preset
   ```

4. 📊 **Use figures in your paper/presentation:**
   - LaTeX: Use PDF files (`\includegraphics{figure.pdf}`)
   - PowerPoint: Use PNG files (high DPI)
   - Web: Use PNG files (medium DPI) or SVG

---

## ✨ SUMMARY

**What you can do now:**

✅ Generate publication-quality figures (300 DPI, vector graphics)
✅ Create presentation-ready visuals (large fonts, bright colors)
✅ Export in multiple formats (PNG, SVG, PDF)
✅ Regenerate from any evaluation results (no re-training needed)
✅ Customize DPI, fonts, and styling

**Just copy and paste the commands above!** 🚀
