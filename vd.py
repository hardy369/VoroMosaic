import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple, Optional
import argparse
import os

class Point:
    """Simple point class for seed points"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other_x: float, other_y: float) -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

class CustomVoronoiMosaic:
    """Custom Voronoi mosaic generator without using external Voronoi libraries"""
    
    def __init__(self, image_path: str):
        """Initialize with an image"""
        self.original_image = Image.open(image_path).convert('RGB')
        self.width, self.height = self.original_image.size
        self.image_array = np.array(self.original_image)
        self.seed_points = []
        self.mosaic_image = None
        
    def resize_image(self, max_size: int = 800) -> None:
        """Resize image to manageable size while maintaining aspect ratio"""
        if max(self.width, self.height) > max_size:
            if self.width > self.height:
                new_width = max_size
                new_height = int(self.height * max_size / self.width)
            else:
                new_height = max_size
                new_width = int(self.width * max_size / self.height)
            
            self.original_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.width, self.height = new_width, new_height
            self.image_array = np.array(self.original_image)
    
    def generate_random_points(self, num_points: int) -> List[Point]:
        """Generate random seed points"""
        points = []
        for _ in range(num_points):
            x = random.uniform(0, self.width - 1)
            y = random.uniform(0, self.height - 1)
            points.append(Point(x, y))
        return points
    
    def generate_poisson_disk_points(self, num_points: int) -> List[Point]:
        """Generate points using Poisson disk sampling for more even distribution"""
        points = []
        min_distance = math.sqrt((self.width * self.height) / num_points) * 0.8
        max_attempts = 30
        
        # Start with a random point
        first_point = Point(random.uniform(0, self.width - 1), 
                           random.uniform(0, self.height - 1))
        points.append(first_point)
        active_list = [0]  # Indices of active points
        
        while active_list and len(points) < num_points:
            # Pick a random active point
            active_idx = random.choice(active_list)
            current_point = points[active_idx]
            
            found_valid = False
            
            for _ in range(max_attempts):
                # Generate point around current point
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(min_distance, 2 * min_distance)
                
                new_x = current_point.x + math.cos(angle) * radius
                new_y = current_point.y + math.sin(angle) * radius
                
                # Check bounds
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    # Check minimum distance to all existing points
                    valid = True
                    for existing_point in points:
                        if existing_point.distance_to(new_x, new_y) < min_distance:
                            valid = False
                            break
                    
                    if valid:
                        new_point = Point(new_x, new_y)
                        points.append(new_point)
                        active_list.append(len(points) - 1)
                        found_valid = True
                        break
            
            if not found_valid:
                active_list.remove(active_idx)
        
        return points
    
    def generate_grid_with_noise_points(self, num_points: int) -> List[Point]:
        """Generate grid-based points with random noise"""
        points = []
        grid_size = int(math.sqrt(num_points))
        cell_width = self.width / grid_size
        cell_height = self.height / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= num_points:
                    break
                
                # Base grid position
                base_x = (i + 0.5) * cell_width
                base_y = (j + 0.5) * cell_height
                
                # Add noise (up to 30% of cell size)
                noise_x = random.uniform(-0.3 * cell_width, 0.3 * cell_width)
                noise_y = random.uniform(-0.3 * cell_height, 0.3 * cell_height)
                
                final_x = max(0, min(self.width - 1, base_x + noise_x))
                final_y = max(0, min(self.height - 1, base_y + noise_y))
                
                points.append(Point(final_x, final_y))
        
        return points
    
    def find_closest_point_index(self, x: int, y: int) -> int:
        """Find the index of the closest seed point to given coordinates"""
        min_distance = float('inf')
        closest_index = 0
        
        for i, point in enumerate(self.seed_points):
            distance = point.distance_to(x, y)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index
    
    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get RGB color of pixel at given coordinates"""
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return tuple(self.image_array[y, x])
    
    def get_average_color_in_radius(self, center_x: int, center_y: int, radius: int) -> Tuple[int, int, int]:
        """Get average color within a radius around the center point"""
        r_sum = g_sum = b_sum = 0
        count = 0
        
        min_x = max(0, center_x - radius)
        max_x = min(self.width - 1, center_x + radius)
        min_y = max(0, center_y - radius)
        max_y = min(self.height - 1, center_y + radius)
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance <= radius:
                    r, g, b = self.get_pixel_color(x, y)
                    r_sum += r
                    g_sum += g
                    b_sum += b
                    count += 1
        
        if count == 0:
            return self.get_pixel_color(center_x, center_y)
        
        return (r_sum // count, g_sum // count, b_sum // count)
    
    def get_dominant_color_in_radius(self, center_x: int, center_y: int, radius: int) -> Tuple[int, int, int]:
        """Get the most frequent color within a radius (with color quantization)"""
        color_counts = {}
        
        min_x = max(0, center_x - radius)
        max_x = min(self.width - 1, center_x + radius)
        min_y = max(0, center_y - radius)
        max_y = min(self.height - 1, center_y + radius)
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance <= radius:
                    r, g, b = self.get_pixel_color(x, y)
                    # Quantize colors to reduce noise
                    quantized = ((r // 16) * 16, (g // 16) * 16, (b // 16) * 16)
                    color_counts[quantized] = color_counts.get(quantized, 0) + 1
        
        if not color_counts:
            return self.get_pixel_color(center_x, center_y)
        
        return max(color_counts.items(), key=lambda x: x[1])[0]
    
    def calculate_point_color(self, point: Point, color_mode: str) -> Tuple[int, int, int]:
        """Calculate color for a seed point based on the specified mode"""
        x, y = int(point.x), int(point.y)
        
        if color_mode == 'center':
            return self.get_pixel_color(x, y)
        elif color_mode == 'average':
            return self.get_average_color_in_radius(x, y, 10)
        elif color_mode == 'dominant':
            return self.get_dominant_color_in_radius(x, y, 15)
        else:
            return self.get_pixel_color(x, y)
    
    def generate_voronoi_mosaic(self, 
                               num_points: int = 300000, 
                               point_distribution: str = 'random',
                               color_mode: str = 'center',
                               show_edges: bool = True) -> Image.Image:
        """Generate the Voronoi mosaic"""
        print(f"Generating {num_points} seed points using {point_distribution} distribution...")
        
        # Generate seed points based on distribution method
        if point_distribution == 'random':
            self.seed_points = self.generate_random_points(num_points)
        elif point_distribution == 'poisson':
            self.seed_points = self.generate_poisson_disk_points(num_points)
        elif point_distribution == 'grid':
            self.seed_points = self.generate_grid_with_noise_points(num_points)
        else:
            raise ValueError("point_distribution must be 'random', 'poisson', or 'grid'")
        
        print(f"Generated {len(self.seed_points)} seed points")
        print(f"Calculating colors for each region using {color_mode} method...")
        
        # Pre-calculate colors for each seed point
        point_colors = []
        for i, point in enumerate(self.seed_points):
            
            color = self.calculate_point_color(point, color_mode)
            point_colors.append(color)
        
        print("Generating Voronoi diagram...")
        
        # Create the mosaic image
        mosaic_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # For each pixel, find closest point and assign its color
        total_pixels = self.width * self.height
        processed = 0
        
        for y in range(self.height):
            for x in range(self.width):
                closest_index = self.find_closest_point_index(x, y)
                mosaic_array[y, x] = point_colors[closest_index]
                
                processed += 1
                        
        self.mosaic_image = Image.fromarray(mosaic_array)
        
        # Add edges if requested
        if show_edges:
            print("Adding cell edges...")
            self.add_voronoi_edges()
        
        print("Voronoi mosaic generation complete!")
        return self.mosaic_image
    
    def add_voronoi_edges(self, edge_color: Tuple[int, int, int] = (255, 255, 255), 
                         edge_thickness: int = 1):
        """Add edges to the Voronoi cells using simple edge detection"""
        if self.mosaic_image is None:
            return
        
        draw = ImageDraw.Draw(self.mosaic_image)
        mosaic_array = np.array(self.mosaic_image)
        
        # Simple edge detection by comparing neighboring pixels
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                current_point = self.find_closest_point_index(x, y)
                right_point = self.find_closest_point_index(x + 1, y)
                down_point = self.find_closest_point_index(x, y + 1)
                
                # If neighboring pixels belong to different cells, draw edge
                if current_point != right_point or current_point != down_point:
                    # Draw a small circle for the edge
                    draw.ellipse([x - edge_thickness//2, y - edge_thickness//2, 
                                 x + edge_thickness//2, y + edge_thickness//2], 
                                fill=edge_color)
    
    def save_results(self, output_dir: str = "voronoi_output", prefix: str = "mosaic"):
        """Save the original and mosaic images"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original (resized if applicable)
        original_path = os.path.join(output_dir, f"{prefix}_original.png")
        self.original_image.save(original_path)
        
        # Save mosaic
        if self.mosaic_image:
            mosaic_path = os.path.join(output_dir, f"{prefix}_voronoi.png")
            self.mosaic_image.save(mosaic_path)
            print(f"Images saved to {output_dir}/")
            return original_path, mosaic_path
        else:
            print("No mosaic generated yet!")
            return original_path, None
    
    def display_results(self):
        """Display original and mosaic side by side using matplotlib"""
        if self.mosaic_image is None:
            print("No mosaic generated yet!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        ax1.imshow(self.original_image)
        ax1.set_title("Original Image", fontsize=14)
        ax1.axis('off')
        
        ax2.imshow(self.mosaic_image)
        ax2.set_title("Voronoi Mosaic", fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def display_seed_points_overlay(self):
        """Display the original image with seed points overlaid"""
        if not self.seed_points:
            print("No seed points generated yet!")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(self.original_image)
        
        # Plot seed points
        x_coords = [point.x for point in self.seed_points]
        y_coords = [point.y for point in self.seed_points]
        
        ax.scatter(x_coords, y_coords, c='red', s=2, alpha=0.7)
        ax.set_title(f"Original Image with {len(self.seed_points)} Seed Points", fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Example usage of the CustomVoronoiMosaic class"""
    parser = argparse.ArgumentParser(description='Generate Voronoi mosaics from images')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--points', type=int, default=300, help='Number of seed points (default: 300)')
    parser.add_argument('--distribution', choices=['random', 'poisson', 'grid'], 
                       default='random', help='Point distribution method (default: random)')
    parser.add_argument('--color-mode', choices=['center', 'average', 'dominant'], 
                       default='center', help='Color sampling mode (default: center)')
    parser.add_argument('--no-edges', action='store_true', help='Disable cell edges')
    parser.add_argument('--max-size', type=int, default=800, help='Maximum image size (default: 800)')
    parser.add_argument('--output-dir', default='voronoi_output', help='Output directory (default: voronoi_output)')
    parser.add_argument('--show-points', action='store_true', help='Display seed points overlay')
    parser.add_argument('--no-display', action='store_true', help='Skip displaying results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        return
    
    # Create mosaic generator
    print(f"Loading image: {args.image_path}")
    mosaic_generator = CustomVoronoiMosaic(args.image_path)
    
    # Resize if needed
    if max(mosaic_generator.width, mosaic_generator.height) > args.max_size:
        print(f"Resizing image to max size {args.max_size}...")
        mosaic_generator.resize_image(args.max_size)
    
    print(f"Image size: {mosaic_generator.width} x {mosaic_generator.height}")
    
    # Generate mosaic
    mosaic_generator.generate_voronoi_mosaic(
        num_points=args.points,
        point_distribution=args.distribution,
        color_mode=args.color_mode,
        show_edges=not args.no_edges
    )
    
    # Save results
    mosaic_generator.save_results(args.output_dir)
    
    # Display results
    if not args.no_display:
        if args.show_points:
            mosaic_generator.display_seed_points_overlay()
        mosaic_generator.display_results()

if __name__ == "__main__":
    main()
    
  