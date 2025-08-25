# VoroMosaic 🎨

**A Custom Voronoi Mosaic Generator Built From Scratch**

Transform your images into stunning geometric mosaics using computational geometry algorithms implemented without external Voronoi libraries.
<img width="1919" height="796" alt="image" src="https://github.com/user-attachments/assets/6a051e97-494e-4a73-8110-f9ecc90731dc" />

### Custom Implementation
- **No External Voronoi Libraries**: Built from scratch using pure computational geometry
- **Multiple Point Distribution Algorithms**: Random, Poisson Disk Sampling, Grid with Noise
- **Advanced Color Sampling**: Center point, average color, and dominant color modes
- **Custom Edge Detection**: Boundary identification without relying on external algorithms

###  Artistic Control
- **Flexible Point Count**: From 50 to 2000+ seed points for different artistic effects
- **Multiple Distribution Patterns**: Create uniform, organic, or structured layouts
- **Color Analysis Modes**: Fine-tune how colors are extracted from original regions
- **Customizable Edges**: Optional cell boundaries with adjustable styling


## 🛠 Installation

### Requirements
```bash
python >= 3.7
numpy
pillow (PIL)
matplotlib
```

### Install Dependencies
```bash
pip install numpy pillow matplotlib
```

### Download
```bash
git clone https://github.com/yourusername/tessellate.git
cd tessellate
```

## 🚀 Quick Start

### Command Line Usage
```bash
# Basic mosaic generation
python tessellate.py your_image.jpg

# Advanced options
python tessellate.py image.jpg --points 500 --distribution poisson --color-mode average

# High resolution with no edges
python tessellate.py image.jpg --points 800 --max-size 1200 --no-edges
```


```bash
python tessellate.py landscape.jpg --points 400 --distribution poisson --color-mode average
```
Perfect for: Nature scenes, architectural photography

```bash
python tessellate.py portrait.jpg --points 600 --distribution grid --color-mode dominant --no-edges
```
Perfect for: Face details, skin tones, artistic portraits


```bash
python tessellate.py abstract.jpg --points 800 --distribution random --color-mode center
```
Perfect for: Colorful compositions, artistic experimentation

## Algorithm Details

### Voronoi Diagram Generation
1. **Seed Point Generation**: Creates points using selected distribution algorithm
2. **Distance Field Calculation**: For each pixel, calculates Euclidean distance to all seeds
3. **Cell Assignment**: Assigns each pixel to its nearest seed point
4. **Color Sampling**: Applies selected color analysis method to determine cell colors
5. **Edge Detection**: Identifies cell boundaries through neighbor comparison

### Mathematical Foundation
```python
# Euclidean distance calculation
distance = √((x₁ - x₂)² + (y₁ - y₂)²)

# Color quantization for dominant color
quantized_color = ((r // 16) * 16, (g // 16) * 16, (b // 16) * 16)

# Poisson disk sampling constraint
min_distance = √(area / num_points) * 0.8
```

## 🐛 Troubleshooting

### Common Issues

**Image too large / Out of memory**
```bash
python tessellate.py image.jpg --max-size 600
```

**Processing too slow**
```bash
python tessellate.py image.jpg --points 200 --distribution random
```

**Colors look washed out**
```bash
python tessellate.py image.jpg --color-mode dominant
```

**Not enough detail**
```bash
python tessellate.py image.jpg --points 800 --no-edges
```

### Error Messages

| Error | Solution |
|-------|----------|
| "Image file not found" | Check file path and extension |
| "Memory error" | Reduce --max-size or --points |
| "Invalid distribution" | Use: random, poisson, or grid |
| "Invalid color mode" | Use: center, average, or dominant |


