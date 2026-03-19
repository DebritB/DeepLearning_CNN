"""
CNN Pipeline Animation - Scrolls left through each operation
Shows: Input -> Convolution -> Pooling -> Flatten -> Dense -> Output

Run with: manim -ql cnn_artistic.py ArtisticNeuralNetwork
"""

from manim import *
import numpy as np
from PIL import Image
import os

# Get paths from environment
IMAGE_PATH = os.environ.get('CNN_IMAGE_PATH', 'temp_input.png')
PREDICTION_CLASS = os.environ.get('CNN_WINNER_CLASS', 'Sonic')

# Animation speed multiplier (higher = slower)
SPEED = 2.0  # 2x slower than before

# Color palette
COLORS = {
    'bg': '#0a0a12',
    'input': '#00ff88',
    'conv': '#ff4466',
    'kernel': '#ffaa00',
    'pool': '#44aaff',
    'flatten': '#aa44ff',
    'dense': '#44ffaa',
    'output': '#ffdd00',
}


def load_image(path, size=10):
    """Load image as grayscale array"""
    try:
        img = Image.open(path).convert('L').resize((size, size), Image.Resampling.LANCZOS)
        return np.array(img) / 255.0
    except:
        return np.random.rand(size, size)


class ArtisticNeuralNetwork(Scene):
    """Pipeline CNN visualization - scrolls left through each operation"""
    
    def construct(self):
        self.camera.background_color = COLORS['bg']
        
        # Load and process image
        self.img = load_image(IMAGE_PATH, size=10)
        self.conv = self._compute_conv(self.img)
        self.pool = self._compute_pool(self.conv)
        self.flat = self.pool.flatten()
        
        # Title
        self._title()
        
        # Pipeline stages - each shifts left as new stage appears
        self._stage_input()
        self._stage_convolution()
        self._stage_pooling()
        self._stage_flatten()
        self._stage_dense()
        self._stage_output_wrong()  # First: Wrong prediction
        self._backprop_animation()  # Backpropagation
        self._forward_again()       # Second forward pass - correct!
    
    def _compute_conv(self, img, k=3):
        """Convolution operation"""
        h, w = img.shape
        out = np.zeros((h-k+1, w-k+1))
        kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
        for i in range(h-k+1):
            for j in range(w-k+1):
                out[i,j] = np.abs(np.sum(img[i:i+k, j:j+k] * kernel))
        return (out - out.min()) / (out.max() - out.min() + 1e-10)
    
    def _compute_pool(self, img, p=2):
        """Max pooling operation"""
        h, w = img.shape
        out = np.zeros((h//p, w//p))
        for i in range(h//p):
            for j in range(w//p):
                out[i,j] = np.max(img[i*p:i*p+p, j*p:j*p+p])
        return out
    
    def _create_grid(self, data, cell_size, color_type, pos=ORIGIN):
        """Create pixel grid"""
        h, w = data.shape
        grid = VGroup()
        for i in range(h):
            for j in range(w):
                v = float(data[i, j])
                c = self._value_to_color(v, color_type)
                
                sq = Square(cell_size, fill_color=c, fill_opacity=0.95,
                           stroke_width=0.5, stroke_color=GRAY)
                sq.move_to([j * cell_size - w * cell_size / 2 + cell_size/2,
                           -i * cell_size + h * cell_size / 2 - cell_size/2, 0])
                grid.add(sq)
        grid.move_to(pos)
        return grid
    
    def _value_to_color(self, v, color_type):
        """Convert value to color based on color type"""
        if color_type == 'gray':
            return interpolate_color(BLACK, WHITE, v)
        elif color_type == 'conv':
            return interpolate_color(BLACK, ManimColor(COLORS['conv']), v)
        elif color_type == 'pool':
            return interpolate_color(BLACK, ManimColor(COLORS['pool']), v)
        else:
            return interpolate_color(BLACK, WHITE, v)
    
    def _title(self):
        """Show title"""
        title = Text("CNN Image Classification", font_size=44, color=WHITE, weight=BOLD)
        sub = Text("Processing Your Image Through Neural Network", font_size=22, color=GRAY)
        sub.next_to(title, DOWN, buff=0.2)
        
        self.play(Write(title), run_time=1.2 * SPEED)
        self.play(Write(sub), run_time=0.8 * SPEED)
        self.wait(0.6 * SPEED)
        self.play(FadeOut(title), FadeOut(sub), run_time=0.5 * SPEED)
    
    def _stage_input(self):
        """Stage 1: Show input image"""
        # Stage label
        stage_label = Text("1. INPUT IMAGE", font_size=28, color=ManimColor(COLORS['input']))
        stage_label.to_edge(UP, buff=0.4)
        self.play(Write(stage_label), run_time=0.6 * SPEED)
        
        # Show actual image
        if os.path.exists(IMAGE_PATH):
            actual_img = ImageMobject(IMAGE_PATH)
            actual_img.set_height(3.5)
        else:
            actual_img = self._create_grid(self.img, 0.3, 'gray')
        
        actual_img.move_to(ORIGIN)
        
        # Glowing border
        border = SurroundingRectangle(actual_img, color=ManimColor(COLORS['input']), 
                                       buff=0.1, stroke_width=3)
        glow = border.copy().set_stroke(width=10, opacity=0.3)
        
        label = Text("Your Uploaded Image", font_size=18, color=ManimColor(COLORS['input']))
        label.next_to(actual_img, DOWN, buff=0.3)
        
        self.play(FadeIn(actual_img), Create(border), Create(glow), Write(label), run_time=0.5 * SPEED)
        self.wait(0.4 * SPEED)
        
        # Shift everything left
        self.input_group = Group(actual_img, border, glow, label)
        self.play(
            self.input_group.animate.scale(0.5).move_to(LEFT * 5.5),
            FadeOut(stage_label),
            run_time=0.5 * SPEED
        )
    
    def _stage_convolution(self):
        """Stage 2: Convolution operation - creates multiple stacked feature maps (3D)"""
        stage_label = Text("2. CONVOLUTION", font_size=28, color=ManimColor(COLORS['conv']))
        stage_label.to_edge(UP, buff=0.4)
        self.play(Write(stage_label), run_time=0.3 * SPEED)
        
        # Pixel grid of input (for showing kernel)
        input_grid = self._create_grid(self.img, 0.28, 'gray', LEFT * 2)
        input_lbl = Text("Input Pixels", font_size=14, color=WHITE)
        input_lbl.next_to(input_grid, DOWN, buff=0.15)
        
        self.play(FadeIn(input_grid), Write(input_lbl), run_time=0.4 * SPEED)
        
        # Kernel
        kernel = VGroup()
        kernel_vals = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        for i in range(3):
            for j in range(3):
                sq = Square(0.28, fill_color=ManimColor(COLORS['kernel']), fill_opacity=0.7,
                           stroke_color=ManimColor(COLORS['kernel']), stroke_width=2)
                val_txt = Text(f"{kernel_vals[i][j]:+d}", font_size=10, color=WHITE)
                sq.move_to([j * 0.28, -i * 0.28, 0])
                val_txt.move_to(sq.get_center())
                kernel.add(VGroup(sq, val_txt))
        
        kernel.move_to(input_grid.get_corner(UL) + RIGHT * 0.42 + DOWN * 0.42)
        kernel_lbl = Text("3×3 Kernel", font_size=12, color=ManimColor(COLORS['kernel']))
        kernel_lbl.next_to(kernel, UP, buff=0.1)
        
        self.play(FadeIn(kernel), Write(kernel_lbl), run_time=0.3 * SPEED)
        
        # Slide kernel across input
        positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for pi, pj in positions:
            target = input_grid[0].get_center() + RIGHT * (pj * 0.28) + DOWN * (pi * 0.28)
            self.play(kernel.animate.move_to(target + RIGHT * 0.28 + DOWN * 0.28), run_time=0.1 * SPEED)
        
        # Arrow
        arrow = Arrow(input_grid.get_right() + RIGHT * 0.2, RIGHT * 1.5 + LEFT * 0.2,
                     color=ManimColor(COLORS['conv']), stroke_width=3)
        self.play(GrowArrow(arrow), run_time=0.2 * SPEED)
        self.play(FadeOut(kernel), FadeOut(kernel_lbl), run_time=0.15 * SPEED)
        
        # Create STACKED feature maps (3D effect) - multiple filters
        num_filters = 6  # Number of feature maps/filters
        stacked_maps = VGroup()
        
        # Create each layer from back to front (so front renders last)
        for f in range(num_filters - 1, -1, -1):
            # Offset for 3D stacking effect
            offset = f * 0.12  # Diagonal offset for depth
            
            # Create slightly different feature maps (simulate different filters)
            np.random.seed(f + 42)
            filter_variation = np.random.randn(*self.conv.shape) * 0.2
            varied_conv = np.clip(self.conv + filter_variation, 0, 1)
            
            layer = self._create_grid(varied_conv, 0.28, 'conv', RIGHT * 2.5)
            layer.shift(UP * offset + RIGHT * offset)
            
            # Back layers are darker/more transparent (obstructed)
            opacity = 0.3 + 0.7 * (1 - f / num_filters)  # Front is 1.0, back fades
            for sq in layer:
                sq.set_fill(opacity=opacity * 0.9)
                sq.set_stroke(opacity=opacity)
            
            stacked_maps.add(layer)
        
        # The front-most layer (last in group) is fully visible
        front_layer = stacked_maps[-1]
        
        # Animate stacked layers appearing one by one from back to front
        self.play(FadeIn(stacked_maps[0]), run_time=0.2 * SPEED)
        for i in range(1, len(stacked_maps)):
            self.play(FadeIn(stacked_maps[i]), run_time=0.1 * SPEED)
        
        # Label
        conv_lbl = Text(f"{num_filters} Feature Maps", font_size=14, color=ManimColor(COLORS['conv']))
        conv_lbl.next_to(stacked_maps, DOWN, buff=0.25)
        depth_lbl = Text("(stacked depth)", font_size=10, color=GRAY)
        depth_lbl.next_to(conv_lbl, DOWN, buff=0.05)
        
        self.play(Write(conv_lbl), Write(depth_lbl), run_time=0.3 * SPEED)
        self.wait(0.3 * SPEED)
        
        # Store for later - keep all stacked maps
        self.conv_stacked = stacked_maps
        self.conv_group = VGroup(stacked_maps, conv_lbl, depth_lbl)
        
        # Shift everything left
        self.play(
            self.input_group.animate.shift(LEFT * 2),
            FadeOut(input_grid), FadeOut(input_lbl), FadeOut(arrow),
            self.conv_group.animate.shift(LEFT * 3.5),
            FadeOut(stage_label),
            run_time=0.5 * SPEED
        )
    
    def _stage_pooling(self):
        """Stage 3: Max Pooling - animate pooling from conv to create max pooled layer"""
        stage_label = Text("3. MAX POOLING", font_size=28, color=ManimColor(COLORS['pool']))
        stage_label.to_edge(UP, buff=0.4)
        self.play(Write(stage_label), run_time=0.3 * SPEED)
        
        # Get the front layer from stacked conv maps
        front_layer = self.conv_stacked[-1]
        num_filters = 6
        
        # Create the pooled output grid structure (empty initially) - just front layer visible first
        pool_cell_size = 0.38
        pool_h, pool_w = self.pool.shape
        pool_start = RIGHT * 3.5
        
        # Pooling window overlay on front conv layer (covers 2x2 cells)
        conv_cell_size = 0.28
        pool_window = Square(conv_cell_size * 2, fill_color=ManimColor(COLORS['pool']), fill_opacity=0.3,
                            stroke_color=ManimColor(COLORS['pool']), stroke_width=3)
        pool_window.move_to(front_layer.get_corner(UL) + RIGHT * conv_cell_size + DOWN * conv_cell_size)
        
        pool_lbl = Text("2×2 Max Pool", font_size=12, color=ManimColor(COLORS['pool']))
        pool_lbl.next_to(pool_window, UP, buff=0.1)
        
        note = Text("(applies to all layers)", font_size=10, color=GRAY)
        note.next_to(pool_lbl, UP, buff=0.05)
        
        # Arrow from conv to pooled area
        arrow = Arrow(self.conv_stacked.get_right() + RIGHT * 0.2, pool_start + LEFT * 0.8,
                     color=ManimColor(COLORS['pool']), stroke_width=3)
        
        self.play(FadeIn(pool_window), Write(pool_lbl), Write(note), GrowArrow(arrow), run_time=0.3 * SPEED)
        
        # Build pooled layer cell by cell as window scans
        # Only show first few cells being built, then speed up
        pooled_cells = VGroup()
        h, w = self.conv.shape
        
        show_detailed = 4  # Show detailed animation for first N cells
        cell_count = 0
        
        for i in range(pool_h):
            for j in range(pool_w):
                cell_count += 1
                # Position pool window on conv layer
                window_target = front_layer.get_corner(UL) + RIGHT * (j * 2 * conv_cell_size + conv_cell_size) + DOWN * (i * 2 * conv_cell_size + conv_cell_size)
                
                # Position for new pooled cell
                cell_pos = pool_start + RIGHT * (j * pool_cell_size - pool_w * pool_cell_size / 2 + pool_cell_size / 2) + \
                          DOWN * (i * pool_cell_size - pool_h * pool_cell_size / 2 + pool_cell_size / 2)
                
                # Get actual max value for this pool region
                val = self.pool[i, j]
                color = self._value_to_color(val, 'pool')
                
                # Create the pooled cell
                pooled_cell = Square(pool_cell_size, fill_color=color, fill_opacity=0.95,
                                    stroke_color=ManimColor(COLORS['pool']), stroke_width=1)
                pooled_cell.move_to(cell_pos)
                
                if cell_count <= show_detailed:
                    # Detailed animation: move window, highlight, create cell
                    self.play(pool_window.animate.move_to(window_target), run_time=0.12 * SPEED)
                    
                    # Flash the window to show "max" extraction
                    self.play(pool_window.animate.set_fill(opacity=0.7), run_time=0.05 * SPEED)
                    
                    # Create a "max" dot that flies to pooled position
                    max_dot = Dot(color=color, radius=0.08).move_to(window_target)
                    self.play(FadeIn(max_dot, scale=1.5), run_time=0.05 * SPEED)
                    self.play(
                        max_dot.animate.move_to(cell_pos),
                        pool_window.animate.set_fill(opacity=0.3),
                        run_time=0.1 * SPEED
                    )
                    self.play(
                        ReplacementTransform(max_dot, pooled_cell),
                        run_time=0.08 * SPEED
                    )
                else:
                    # Fast animation for remaining cells
                    self.play(
                        pool_window.animate.move_to(window_target),
                        FadeIn(pooled_cell, scale=0.8),
                        run_time=0.03 * SPEED
                    )
                
                pooled_cells.add(pooled_cell)
        
        self.play(FadeOut(pool_window), FadeOut(pool_lbl), FadeOut(note), run_time=0.15 * SPEED)
        
        # Now create the stacked layers behind the front pooled layer
        stacked_pool = VGroup()
        
        # Add back layers first (so they render behind)
        for f in range(num_filters - 2, -1, -1):
            offset = (num_filters - 1 - f) * 0.1
            
            # Varied pooling results for this filter
            np.random.seed(f + 100)
            pool_variation = np.random.rand(*self.pool.shape) * 0.2
            varied_pool = np.clip(self.pool + pool_variation - 0.1, 0, 1)
            
            layer = self._create_grid(varied_pool, pool_cell_size, 'pool', pool_start)
            layer.shift(UP * offset + RIGHT * offset)
            
            opacity = 0.3 + 0.7 * (1 - (num_filters - 1 - f) / num_filters)
            for sq in layer:
                sq.set_fill(opacity=opacity * 0.8)
                sq.set_stroke(opacity=opacity)
            
            stacked_pool.add(layer)
        
        # Animate back layers appearing behind
        back_layers_lbl = Text("+ 5 more layers", font_size=10, color=GRAY)
        back_layers_lbl.next_to(pooled_cells, UR, buff=0.15)
        
        self.play(FadeIn(stacked_pool, shift=UP * 0.1 + RIGHT * 0.1), Write(back_layers_lbl), run_time=0.3 * SPEED)
        
        # Move front layer (pooled_cells) to proper position in stack
        front_offset = (num_filters - 1) * 0.1
        self.play(pooled_cells.animate.shift(DOWN * front_offset + LEFT * front_offset), run_time=0.2 * SPEED)
        
        # Combine into final stack
        stacked_pool.add(pooled_cells)
        
        pool_out_lbl = Text(f"{num_filters} Pooled Maps", font_size=14, color=ManimColor(COLORS['pool']))
        pool_out_lbl.next_to(stacked_pool, DOWN, buff=0.2)
        size_lbl = Text(f"({pool_h}×{pool_w} each)", font_size=10, color=GRAY)
        size_lbl.next_to(pool_out_lbl, DOWN, buff=0.05)
        
        self.play(
            FadeOut(back_layers_lbl),
            Write(pool_out_lbl),
            Write(size_lbl),
            run_time=0.3 * SPEED
        )
        self.wait(0.2 * SPEED)
        
        # Store for flatten stage
        self.pool_stacked = stacked_pool
        self.pool_group = VGroup(stacked_pool, pool_out_lbl, size_lbl)
        
        # Shift left
        self.play(
            self.input_group.animate.shift(LEFT * 2),
            self.conv_group.animate.shift(LEFT * 2),
            FadeOut(arrow),
            self.pool_group.animate.shift(LEFT * 2),
            FadeOut(stage_label),
            run_time=0.5 * SPEED
        )
    
    def _stage_flatten(self):
        """Stage 4: Flatten - show all 3D stacked layers flattening into 1D"""
        stage_label = Text("4. FLATTEN", font_size=28, color=ManimColor(COLORS['flatten']))
        stage_label.to_edge(UP, buff=0.4)
        self.play(Write(stage_label), run_time=0.3 * SPEED)
        
        # Arrow showing transformation
        arrow = Arrow(self.pool_stacked.get_right() + RIGHT * 0.3, RIGHT * 1.5,
                     color=ManimColor(COLORS['flatten']), stroke_width=3)
        arrow_txt = Text("Flatten All", font_size=14, color=ManimColor(COLORS['flatten']))
        arrow_txt.next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), Write(arrow_txt), run_time=0.3 * SPEED)
        
        # Total flattened values = all pixels from all layers
        total_values = len(self.flat) * 6  # 6 filters
        num_show = min(16, total_values)
        
        # Create flying dots from ALL stacked layers
        flying_dots = VGroup()
        dot_idx = 0
        
        # Collect dots from each layer (front to back)
        for layer_idx, layer in enumerate(reversed(list(self.pool_stacked))):
            layer_squares = list(layer)
            for sq_idx, sq in enumerate(layer_squares):
                if dot_idx >= num_show:
                    break
                dot = Circle(0.08, fill_color=sq.get_fill_color(), fill_opacity=0.9,
                            stroke_color=ManimColor(COLORS['flatten']), stroke_width=1)
                dot.move_to(sq.get_center())
                flying_dots.add(dot)
                dot_idx += 1
            if dot_idx >= num_show:
                break
        
        self.play(FadeIn(flying_dots), run_time=0.2 * SPEED)
        
        # Animate dots flying to 1D column positions
        anims = []
        for i, dot in enumerate(flying_dots):
            target_pos = RIGHT * 3 + DOWN * (i * 0.22 - num_show * 0.11 + 0.11)
            anims.append(dot.animate.move_to(target_pos))
        
        self.play(
            *anims,
            self.pool_stacked.animate.set_opacity(0.2),
            run_time=0.8 * SPEED,
            rate_func=smooth
        )
        
        # Create final styled neurons
        neurons = VGroup()
        for i in range(num_show):
            c = flying_dots[i].get_fill_color() if i < len(flying_dots) else ManimColor(COLORS['flatten'])
            neuron = Circle(0.09, fill_color=c, fill_opacity=0.9,
                          stroke_color=ManimColor(COLORS['flatten']), stroke_width=1)
            neuron.move_to(RIGHT * 3 + DOWN * (i * 0.22 - num_show * 0.11 + 0.11))
            neurons.add(neuron)
        
        self.play(
            *[Transform(flying_dots[i], neurons[i]) for i in range(len(flying_dots))],
            run_time=0.3 * SPEED
        )
        
        # Add dots to show there's more
        dots_txt = Text("...", font_size=16, color=ManimColor(COLORS['flatten']))
        dots_txt.next_to(flying_dots, DOWN, buff=0.08)
        
        total_txt = Text(f"({total_values} values)", font_size=10, color=GRAY)
        total_txt.next_to(dots_txt, DOWN, buff=0.05)
        
        flat_lbl = Text("1D Vector", font_size=14, color=ManimColor(COLORS['flatten']))
        flat_lbl.next_to(total_txt, DOWN, buff=0.1)
        
        self.play(FadeIn(dots_txt), Write(total_txt), Write(flat_lbl), run_time=0.3 * SPEED)
        self.wait(0.2 * SPEED)
        
        # Store and shift left - give more space to avoid overlap
        self.flat_group = VGroup(flying_dots, dots_txt, total_txt, flat_lbl)
        self.play(
            self.input_group.animate.shift(LEFT * 2.5),
            self.conv_group.animate.shift(LEFT * 2.5),
            self.pool_group.animate.shift(LEFT * 2.5),
            FadeOut(arrow), FadeOut(arrow_txt),
            self.flat_group.animate.move_to(RIGHT * 1.5 + DOWN * 0.2),
            FadeOut(stage_label),
            run_time=0.5 * SPEED
        )
    
    def _stage_dense(self):
        """Stage 5: Dense layers"""
        stage_label = Text("5. DENSE LAYERS", font_size=28, color=ManimColor(COLORS['dense']))
        stage_label.to_edge(UP, buff=0.4)
        self.play(Write(stage_label), run_time=0.3 * SPEED)
        
        # Dense layer neurons - position to the right of flat_group
        dense1 = VGroup()
        for i in range(6):
            neuron = Circle(0.15, fill_color=ManimColor(COLORS['dense']), fill_opacity=0.7,
                          stroke_color=ManimColor(COLORS['dense']), stroke_width=1)
            neuron.move_to(RIGHT * 4 + DOWN * (i * 0.35 - 0.875))
            dense1.add(neuron)
        
        dense_lbl = Text("Hidden", font_size=12, color=ManimColor(COLORS['dense']))
        dense_lbl.next_to(dense1, DOWN, buff=0.15)
        
        # Connections from flatten to dense
        connections = VGroup()
        flat_neurons = [n for n in self.flat_group[0] if isinstance(n, Circle)]
        for fn in flat_neurons[:6]:
            for dn in dense1:
                line = Line(fn.get_right(), dn.get_left(), 
                           stroke_color=GRAY, stroke_width=0.3, stroke_opacity=0.4)
                connections.add(line)
        
        self.play(Create(connections), run_time=0.3 * SPEED)
        self.play(LaggedStart(*[FadeIn(n, scale=0.5) for n in dense1], lag_ratio=0.05), run_time=0.3 * SPEED)
        self.play(Write(dense_lbl), run_time=0.2 * SPEED)
        self.wait(0.2 * SPEED)
        
        # Shift left - larger shift to make room for output
        self.dense_group = VGroup(dense1, dense_lbl, connections)
        self.play(
            self.input_group.animate.shift(LEFT * 2),
            self.conv_group.animate.shift(LEFT * 2),
            self.pool_group.animate.shift(LEFT * 2),
            self.flat_group.animate.shift(LEFT * 2),
            self.dense_group.animate.shift(LEFT * 2),
            FadeOut(stage_label),
            run_time=0.5 * SPEED
        )
    
    def _stage_output_wrong(self):
        """Stage 6: Output classification - WRONG first time"""
        stage_label = Text("6. CLASSIFICATION", font_size=28, color=ManimColor(COLORS['output']))
        stage_label.to_edge(UP, buff=0.4)
        self.play(Write(stage_label), run_time=0.3 * SPEED)
        
        # Output neurons with class labels - WRONG class selected
        classes = ["Class A", "Class B", PREDICTION_CLASS, "Class D"]  # Correct is at index 2
        scores = [0.65, 0.20, 0.10, 0.05]  # Wrong! Class A selected
        
        outputs = VGroup()
        labels = VGroup()
        connections = VGroup()
        
        for i, (cls, score) in enumerate(zip(classes, scores)):
            is_selected = (i == 0)  # Class A is wrongly selected
            
            neuron = Circle(
                0.25 if is_selected else 0.18,
                fill_color=ManimColor(COLORS['output']) if is_selected else GRAY,
                fill_opacity=0.9 if is_selected else 0.5,
                stroke_color=ManimColor(COLORS['output']) if is_selected else GRAY,
                stroke_width=3 if is_selected else 1
            )
            neuron.move_to(RIGHT * 3.5 + DOWN * (i * 0.7 - 1.05))
            outputs.add(neuron)
            
            lbl = Text(f"{cls}\n{score*100:.0f}%", font_size=12 if is_selected else 10,
                      color=ManimColor(COLORS['output']) if is_selected else WHITE)
            lbl.next_to(neuron, RIGHT, buff=0.15)
            labels.add(lbl)
            
            # Connect from dense
            for dn in self.dense_group[0]:
                line = Line(dn.get_right(), neuron.get_left(),
                           stroke_color=ManimColor(COLORS['output']) if is_selected else GRAY,
                           stroke_width=1 if is_selected else 0.3,
                           stroke_opacity=0.8 if is_selected else 0.3)
                connections.add(line)
        
        self.play(Create(connections), run_time=0.3 * SPEED)
        self.play(
            LaggedStart(*[FadeIn(o, scale=0.5) for o in outputs], lag_ratio=0.08),
            LaggedStart(*[Write(l) for l in labels], lag_ratio=0.08),
            run_time=0.5 * SPEED
        )
        
        # Wrong prediction with X
        wrong_box = SurroundingRectangle(VGroup(outputs[0], labels[0]), 
                                          color=RED, buff=0.1, stroke_width=2)
        
        self.play(Create(wrong_box), run_time=0.3 * SPEED)
        
        # Show error
        error_text = Text("✗ WRONG PREDICTION!", font_size=28, color=RED, weight=BOLD)
        error_text.to_edge(DOWN, buff=0.5)
        
        expected = Text(f"Expected: {PREDICTION_CLASS}", font_size=18, color=GREEN)
        expected.next_to(error_text, UP, buff=0.15)
        
        self.play(Write(error_text), Write(expected), run_time=0.4 * SPEED)
        self.wait(0.5 * SPEED)
        
        # Store for backprop
        self.output_neurons = outputs
        self.output_labels = labels
        self.output_connections = connections
        self.wrong_box = wrong_box
        self.error_text = error_text
        self.expected_text = expected
        self.stage_label = stage_label
        self.classes = classes
    
    def _backprop_animation(self):
        """Animate backpropagation - slide back through network layer by layer"""
        # Change stage label
        backprop_label = Text("BACKPROPAGATION", font_size=28, color=RED)
        backprop_label.to_edge(UP, buff=0.4)
        self.play(Transform(self.stage_label, backprop_label), run_time=0.3 * SPEED)
        
        # Show loss/error signal at output
        loss_text = Text("Loss = Error²", font_size=16, color=RED)
        loss_text.next_to(self.output_neurons[0], UP, buff=0.3)
        self.play(Write(loss_text), run_time=0.25 * SPEED)
        self.wait(0.3 * SPEED)
        
        # Create error gradient indicator that will travel backward
        error_grad = VGroup(
            Circle(0.15, fill_color=RED, fill_opacity=0.8, stroke_color=RED, stroke_width=2),
            Text("∇", font_size=18, color=WHITE)
        )
        error_grad[1].move_to(error_grad[0].get_center())
        error_grad.move_to(self.output_neurons[0].get_center())
        
        grad_label = Text("Error Gradient", font_size=12, color=RED)
        grad_label.next_to(error_grad, DOWN, buff=0.1)
        
        self.play(FadeIn(error_grad), Write(grad_label), run_time=0.3 * SPEED)
        
        # === SLIDE 1: Backprop through Output → Dense connections ===
        # Shift everything RIGHT to make room on left
        self.play(
            self.input_group.animate.shift(RIGHT * 2),
            self.conv_group.animate.shift(RIGHT * 2),
            self.pool_group.animate.shift(RIGHT * 2),
            self.flat_group.animate.shift(RIGHT * 2),
            self.dense_group.animate.shift(RIGHT * 2),
            VGroup(self.output_neurons, self.output_labels, self.output_connections).animate.shift(RIGHT * 2),
            self.wrong_box.animate.shift(RIGHT * 2),
            error_grad.animate.shift(RIGHT * 2),
            grad_label.animate.shift(RIGHT * 2),
            loss_text.animate.shift(RIGHT * 2),
            FadeOut(self.error_text),
            FadeOut(self.expected_text),
            run_time=0.5 * SPEED
        )
        
        phase1 = Text("Step 1: Update Output → Hidden Weights", font_size=16, color=ORANGE)
        phase1.to_edge(DOWN, buff=0.4)
        self.play(Write(phase1), run_time=0.25 * SPEED)
        
        # Animate gradient flowing back through connections
        self.play(
            error_grad.animate.move_to(self.dense_group[0].get_center()),
            grad_label.animate.move_to(self.dense_group[0].get_center() + DOWN * 0.5),
            run_time=0.6 * SPEED
        )
        
        # Flash connections and update weights
        w_old = Text("w = 0.3", font_size=14, color=GRAY)
        w_new = Text("w = 0.7", font_size=14, color=ORANGE)
        w_old.next_to(self.dense_group[0], UP, buff=0.2)
        w_new.move_to(w_old.get_center())
        
        self.play(
            *[n.animate.set_fill(color=ORANGE, opacity=1) for n in self.dense_group[0]],
            FadeIn(w_old),
            run_time=0.3 * SPEED
        )
        self.play(Transform(w_old, w_new), run_time=0.25 * SPEED)
        self.play(
            *[n.animate.set_fill(color=ManimColor(COLORS['dense']), opacity=0.7) for n in self.dense_group[0]],
            run_time=0.25 * SPEED
        )
        
        self.play(FadeOut(phase1), FadeOut(w_old), FadeOut(loss_text), FadeOut(self.wrong_box), run_time=0.2 * SPEED)
        
        # === SLIDE 2: Backprop through Dense → Flatten ===
        phase2 = Text("Step 2: Backprop through Flatten Layer", font_size=16, color=PURPLE)
        phase2.to_edge(DOWN, buff=0.4)
        self.play(Write(phase2), run_time=0.25 * SPEED)
        
        # Shift everything right again, gradient moves left to flatten
        self.play(
            self.input_group.animate.shift(RIGHT * 1.5),
            self.conv_group.animate.shift(RIGHT * 1.5),
            self.pool_group.animate.shift(RIGHT * 1.5),
            self.flat_group.animate.shift(RIGHT * 1.5),
            self.dense_group.animate.shift(RIGHT * 1.5),
            VGroup(self.output_neurons, self.output_labels, self.output_connections).animate.shift(RIGHT * 1.5),
            error_grad.animate.move_to(self.flat_group[0].get_center()),
            grad_label.animate.move_to(self.flat_group[0].get_center() + DOWN * 1.2),
            run_time=0.5 * SPEED
        )
        
        # Flash flatten neurons
        flat_neurons = [n for n in self.flat_group[0] if isinstance(n, Circle)]
        self.play(
            *[n.animate.set_fill(color=PURPLE, opacity=1) for n in flat_neurons[:8]],
            run_time=0.25 * SPEED
        )
        self.play(
            *[n.animate.set_fill(color=ManimColor(COLORS['flatten']), opacity=0.9) for n in flat_neurons[:8]],
            run_time=0.2 * SPEED
        )
        
        self.play(FadeOut(phase2), run_time=0.15 * SPEED)
        
        # === SLIDE 3: Backprop through Flatten → Pool ===
        phase3 = Text("Step 3: Reshape Gradients to 2D", font_size=16, color=ManimColor(COLORS['pool']))
        phase3.to_edge(DOWN, buff=0.4)
        self.play(Write(phase3), run_time=0.25 * SPEED)
        
        # Gradient moves to pool layer
        self.play(
            self.input_group.animate.shift(RIGHT * 1.5),
            self.conv_group.animate.shift(RIGHT * 1.5),
            self.pool_group.animate.shift(RIGHT * 1.5),
            self.flat_group.animate.shift(RIGHT * 1.5),
            self.dense_group.animate.shift(RIGHT * 1.5),
            VGroup(self.output_neurons, self.output_labels, self.output_connections).animate.shift(RIGHT * 1.5),
            error_grad.animate.move_to(self.pool_stacked[-1].get_center()),
            grad_label.animate.move_to(self.pool_stacked[-1].get_center() + DOWN * 1),
            run_time=0.5 * SPEED
        )
        
        # Flash pool layers
        for layer in self.pool_stacked:
            self.play(
                *[sq.animate.set_fill(opacity=1) for sq in layer],
                run_time=0.1 * SPEED
            )
        
        self.play(FadeOut(phase3), run_time=0.15 * SPEED)
        
        # === SLIDE 4: Backprop through Pool → Conv (expand gradients) ===
        phase4 = Text("Step 4: Expand Gradients (Unpooling)", font_size=16, color=ManimColor(COLORS['conv']))
        phase4.to_edge(DOWN, buff=0.4)
        self.play(Write(phase4), run_time=0.25 * SPEED)
        
        # Gradient moves to conv layer
        self.play(
            self.input_group.animate.shift(RIGHT * 1.5),
            self.conv_group.animate.shift(RIGHT * 1.5),
            self.pool_group.animate.shift(RIGHT * 1.5),
            self.flat_group.animate.shift(RIGHT * 1.5),
            self.dense_group.animate.shift(RIGHT * 1.5),
            VGroup(self.output_neurons, self.output_labels, self.output_connections).animate.shift(RIGHT * 1.5),
            error_grad.animate.move_to(self.conv_stacked[-1].get_center()),
            grad_label.animate.move_to(self.conv_stacked[-1].get_center() + DOWN * 1.2),
            run_time=0.5 * SPEED
        )
        
        # Flash conv layers
        for layer in self.conv_stacked:
            self.play(
                *[sq.animate.set_fill(opacity=1) for sq in layer[:16]],  # Just first 16 cells
                run_time=0.08 * SPEED
            )
        
        self.play(FadeOut(phase4), run_time=0.15 * SPEED)
        
        # === SLIDE 5: Update Convolution Filters ===
        phase5 = Text("Step 5: Update Convolution Filter Weights", font_size=16, color=YELLOW)
        phase5.to_edge(DOWN, buff=0.4)
        self.play(Write(phase5), run_time=0.25 * SPEED)
        
        # Move gradient indicator off, fade out network
        self.play(
            FadeOut(error_grad),
            FadeOut(grad_label),
            self.input_group.animate.set_opacity(0.15),
            self.conv_group.animate.set_opacity(0.15),
            self.pool_group.animate.set_opacity(0.15),
            self.flat_group.animate.set_opacity(0.15),
            self.dense_group.animate.set_opacity(0.15),
            self.output_neurons.animate.set_opacity(0.15),
            self.output_labels.animate.set_opacity(0.15),
            self.output_connections.animate.set_opacity(0.15),
            run_time=0.3 * SPEED
        )
        
        # Create enlarged kernel to show filter values updating
        kernel_zoom = VGroup()
        old_vals = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        new_vals = [[2, 1, -1], [3, 0, -3], [2, 1, -2]]
        
        for i in range(3):
            for j in range(3):
                sq = Square(0.6, fill_color=ManimColor(COLORS['kernel']), fill_opacity=0.8,
                           stroke_color=YELLOW, stroke_width=3)
                sq.move_to([j * 0.65 - 0.65, -i * 0.65 + 0.65, 0])
                old_txt = Text(f"{old_vals[i][j]:+.1f}", font_size=16, color=WHITE)
                old_txt.move_to(sq.get_center())
                kernel_zoom.add(VGroup(sq, old_txt))
        
        kernel_zoom.move_to(ORIGIN)
        zoom_label = Text("Convolution Filter #1", font_size=18, color=YELLOW)
        zoom_label.next_to(kernel_zoom, UP, buff=0.25)
        
        self.play(FadeIn(kernel_zoom), Write(zoom_label), run_time=0.3 * SPEED)
        
        # Animate each filter value updating
        for idx, cell in enumerate(kernel_zoom):
            i, j = idx // 3, idx % 3
            old_val = old_vals[i][j]
            new_val = new_vals[i][j]
            if old_val != new_val:
                new_txt = Text(f"{new_val:+.1f}", font_size=16, color=YELLOW)
                new_txt.move_to(cell[1].get_center())
                self.play(
                    cell[0].animate.set_fill(color=YELLOW, opacity=1),
                    Transform(cell[1], new_txt),
                    run_time=0.2 * SPEED
                )
                self.play(
                    cell[0].animate.set_fill(color=ManimColor(COLORS['kernel']), opacity=0.8),
                    run_time=0.12 * SPEED
                )
        
        update_done = Text("✓ All Weights Updated!", font_size=22, color=GREEN)
        update_done.next_to(kernel_zoom, DOWN, buff=0.35)
        self.play(Write(update_done), run_time=0.3 * SPEED)
        self.wait(0.4 * SPEED)
        
        # Fade out zoomed view and phase label
        self.play(
            FadeOut(kernel_zoom), FadeOut(zoom_label), FadeOut(update_done), FadeOut(phase5),
            run_time=0.3 * SPEED
        )
        
        # Restore opacity and slide everything back to original positions (LEFT)
        total_shift = RIGHT * (2 + 1.5 + 1.5 + 1.5)  # Total we shifted right
        self.play(
            self.input_group.animate.set_opacity(1).shift(LEFT * 8),
            self.conv_group.animate.set_opacity(1).shift(LEFT * 8),
            self.pool_group.animate.set_opacity(1).shift(LEFT * 8),
            self.flat_group.animate.set_opacity(1).shift(LEFT * 8),
            self.dense_group.animate.set_opacity(1).shift(LEFT * 8),
            self.output_neurons.animate.set_opacity(1).shift(LEFT * 8),
            self.output_labels.animate.set_opacity(1).shift(LEFT * 8),
            self.output_connections.animate.set_opacity(1).shift(LEFT * 8),
            run_time=0.6 * SPEED
        )
        
        self.play(FadeOut(self.stage_label), run_time=0.2 * SPEED)
    
    def _forward_again(self):
        """Second forward pass with updated weights - CORRECT this time!"""
        # New forward pass label
        forward_label = Text("FORWARD PASS #2 (Updated Weights)", font_size=28, color=GREEN)
        forward_label.to_edge(UP, buff=0.4)
        self.play(Write(forward_label), run_time=0.3 * SPEED)
        
        # Create data flow indicator
        data_flow = VGroup(
            Circle(0.12, fill_color=GREEN, fill_opacity=0.9, stroke_color=WHITE, stroke_width=2),
            Text("→", font_size=14, color=WHITE)
        )
        data_flow[1].move_to(data_flow[0].get_center())
        data_flow.move_to(self.input_group.get_center())
        
        flow_label = Text("Data", font_size=11, color=GREEN)
        flow_label.next_to(data_flow, DOWN, buff=0.08)
        
        self.play(FadeIn(data_flow), Write(flow_label), run_time=0.25 * SPEED)
        
        # === Flow through Input → Conv ===
        self.play(
            data_flow.animate.move_to(self.conv_stacked[-1].get_center()),
            flow_label.animate.move_to(self.conv_stacked[-1].get_center() + DOWN * 0.8),
            *[sq.animate.set_fill(opacity=1) for sq in self.conv_stacked[-1]],
            run_time=0.4 * SPEED
        )
        
        # === Flow through Conv → Pool ===
        self.play(
            data_flow.animate.move_to(self.pool_stacked[-1].get_center()),
            flow_label.animate.move_to(self.pool_stacked[-1].get_center() + DOWN * 0.6),
            run_time=0.35 * SPEED
        )
        
        # === Flow through Pool → Flatten ===
        flat_neurons = [n for n in self.flat_group[0] if isinstance(n, Circle)]
        self.play(
            data_flow.animate.move_to(self.flat_group[0].get_center()),
            flow_label.animate.move_to(self.flat_group[0].get_center() + DOWN * 1.5),
            *[n.animate.set_fill(opacity=1) for n in flat_neurons[:8]],
            run_time=0.35 * SPEED
        )
        
        # === Flow through Flatten → Dense ===
        self.play(
            data_flow.animate.move_to(self.dense_group[0].get_center()),
            flow_label.animate.move_to(self.dense_group[0].get_center() + DOWN * 0.8),
            *[n.animate.set_fill(color=GREEN, opacity=1) for n in self.dense_group[0]],
            run_time=0.35 * SPEED
        )
        self.play(
            *[n.animate.set_fill(color=ManimColor(COLORS['dense']), opacity=0.8) for n in self.dense_group[0]],
            run_time=0.2 * SPEED
        )
        
        # === Flow to Output - CORRECT prediction ===
        self.play(
            data_flow.animate.move_to(self.output_neurons[2].get_center()),  # Correct class at index 2
            flow_label.animate.move_to(self.output_neurons[2].get_center() + DOWN * 0.5),
            run_time=0.35 * SPEED
        )
        
        self.play(FadeOut(data_flow), FadeOut(flow_label), run_time=0.2 * SPEED)
        
        # Now update output with CORRECT prediction
        correct_idx = 2  # PREDICTION_CLASS is at index 2
        new_scores = [0.05, 0.03, 0.90, 0.02]  # Index 2 now highest
        
        # Animate output neurons changing
        for i, (neuron, label) in enumerate(zip(self.output_neurons, self.output_labels)):
            is_winner = (i == correct_idx)
            new_color = ManimColor(COLORS['output']) if is_winner else GRAY
            new_opacity = 0.95 if is_winner else 0.4
            
            # Create new label with updated scores
            new_lbl = Text(f"{self.classes[i]}\n{new_scores[i]*100:.0f}%", 
                          font_size=14 if is_winner else 10,
                          color=ManimColor(COLORS['output']) if is_winner else WHITE)
            new_lbl.next_to(neuron, RIGHT, buff=0.15)
            
            if is_winner:
                # Dramatic grow for winner
                self.play(
                    neuron.animate.scale(1.5).set_fill(color=new_color, opacity=new_opacity)
                                   .set_stroke(color=GREEN, width=4),
                    Transform(label, new_lbl),
                    Flash(neuron.get_center(), color=GREEN, line_length=0.2),
                    run_time=0.4 * SPEED
                )
            else:
                # Shrink losers
                self.play(
                    neuron.animate.scale(0.6).set_fill(color=new_color, opacity=new_opacity)
                                   .set_stroke(color=GRAY, width=1),
                    Transform(label, new_lbl),
                    run_time=0.12 * SPEED
                )
        
        # Winner highlight
        winner_box = SurroundingRectangle(
            VGroup(self.output_neurons[correct_idx], self.output_labels[correct_idx]), 
            color=GREEN, buff=0.12, stroke_width=3
        )
        crown = Text("👑", font_size=30)
        crown.next_to(self.output_neurons[correct_idx], UP, buff=0.08)
        
        self.play(Create(winner_box), FadeIn(crown, scale=1.5), run_time=0.35 * SPEED)
        
        # Final prediction
        result = Text(f"Prediction: {PREDICTION_CLASS}", font_size=34, 
                     color=GREEN, weight=BOLD)
        result.to_edge(DOWN, buff=0.5)
        
        self.play(
            Write(result), 
            Flash(self.output_neurons[correct_idx].get_center(), color=GREEN, line_length=0.4),
            run_time=0.45 * SPEED
        )
        
        # Success message
        success = Text(f"✓ Correctly Identified as {PREDICTION_CLASS}!", font_size=26, color=GREEN)
        success.next_to(result, UP, buff=0.2)
        
        learn_msg = Text("Neural Network Learned from Mistake!", font_size=18, color=YELLOW)
        learn_msg.next_to(forward_label, DOWN, buff=0.15)
        
        self.play(Write(success), Write(learn_msg), run_time=0.4 * SPEED)
        
        self.wait(1.5 * SPEED)


