# Blog Visual Assets Guide

This directory is organized so you can easily drop in screenshots, charts, and photos for your Minimalist EDA blog.

## Directory Layout
* `author/` - Drop an author photo here (e.g. `portrait.jpg` or `avatar.png`).
* `charts/` - Drop your Python/Seaborn/Matplotlib exported charts here (e.g., boxplots, histograms, scatter plots).
* `screenshots/` - Drop screenshots of your Jupyter notebooks, messy CSVs, or DATAVIS experiments here.

## How to Embed an Image in Your Blog (`index.html`)

Copy and paste this snippet into any chapter in `index.html`:

```html
<figure class="baker-image-card">
  <img src="assets/images/charts/your_image_name.png" alt="Descriptive title" loading="lazy">
  <figcaption>Figure X: Add a friendly, 1-2 sentence caption explaining what the reader is looking at.</figcaption>
</figure>
```

### Pre-configured Image Card CSS Features:
* Clean rounded corners (`border-radius: 12px`)
* Subtle warm drop shadow matching the Minimalist Baker aesthetic
* Soft cream caption bar (`<figcaption>`) with italicized styling
* Responsive auto-scaling (scales smoothly on mobile and desktop)
