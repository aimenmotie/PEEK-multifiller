"""
microstructure.py
Microstructure generation module for creating 2D RVEs of PEEK composites.
Implements random sequential adsorption algorithm for particle placement.
Provides visualization for both single and hybrid filler systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


class MicrostructureGenerator:
    """
    Generates 2D representative volume elements (RVEs) with random filler distributions.
    Supports single-filler and hybrid (multi-filler) microstructures.
    """
    
    def __init__(self, rve_size=10.0):
        """
        Parameters:
        -----------
        rve_size : float
            Size of square RVE in micrometers
        """
        self.rve_size = rve_size
        self.random_seed = 42  # For reproducibility
        
    def set_seed(self, seed):
        """Set random seed for reproducible microstructures"""
        self.random_seed = seed
        np.random.seed(seed)
        
    def random_sequential_adsorption(self, num_particles, particle_radius, 
                                      aspect_ratio=1.0, max_attempts=10000, 
                                      min_distance_factor=2.0):
        """
        Place particles using random sequential adsorption algorithm
        
        Parameters:
        -----------
        num_particles : int
            Number of particles to place
        particle_radius : float
            Base radius of particles
        aspect_ratio : float
            Aspect ratio for non-spherical particles
        max_attempts : int
            Maximum placement attempts per particle
        min_distance_factor : float
            Minimum distance between particle centers as multiple of radius
        
        Returns:
        --------
        tuple : (positions, orientations) lists
        """
        np.random.seed(self.random_seed)
        positions = []
        orientations = []  # For non-spherical particles
        
        # Calculate effective particle size for overlap checking
        if aspect_ratio > 1.1:
            # For non-spherical, use average of dimensions
            eff_radius = particle_radius * (1 + aspect_ratio) / 2
        else:
            eff_radius = particle_radius
        
        for i in range(num_particles):
            attempts = 0
            placed = False
            
            while not placed and attempts < max_attempts:
                # Generate random position with boundary padding
                padding = eff_radius * 1.5
                x = np.random.uniform(padding, self.rve_size - padding)
                y = np.random.uniform(padding, self.rve_size - padding)
                
                # Check overlap with existing particles
                overlap = False
                for pos in positions:
                    dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if dist < min_distance_factor * eff_radius:
                        overlap = True
                        break
                
                if not overlap:
                    positions.append((x, y))
                    if aspect_ratio > 1.1:
                        # For non-spherical, add random orientation
                        orientations.append(np.random.uniform(0, np.pi))
                    else:
                        orientations.append(0)
                    placed = True
                
                attempts += 1
            
            if not placed:
                print(f"  Warning: Could only place {i} out of {num_particles} particles")
                break
        
        return positions, orientations
    
    def get_particle_params(self, filler_type):
        """
        Get visualization parameters for a filler type
        
        Parameters:
        -----------
        filler_type : str
            Name of the filler
        
        Returns:
        --------
        dict : Parameters for drawing the filler
        """
        params = {
            'Carbon Fiber': {
                'shape': 'ellipse',
                'aspect_ratio': 5.0,
                'base_radius': 0.2,
                'color': 'gray',
                'alpha': 0.7,
                'edgecolor': 'black',
                'linewidth': 1
            },
            'Graphene Nanoplatelets': {
                'shape': 'rectangle',
                'aspect_ratio': 3.0,
                'base_radius': 0.4,
                'color': 'black',
                'alpha': 0.5,
                'edgecolor': 'darkgray',
                'linewidth': 1
            },
            'Glass Fiber': {
                'shape': 'ellipse',
                'aspect_ratio': 4.0,
                'base_radius': 0.25,
                'color': 'lightblue',
                'alpha': 0.6,
                'edgecolor': 'blue',
                'linewidth': 1
            },
            'Carbon Nanotubes': {
                'shape': 'line',
                'aspect_ratio': 10.0,
                'base_radius': 0.1,
                'color': 'red',
                'alpha': 0.8,
                'edgecolor': 'darkred',
                'linewidth': 2
            }
        }
        
        # Default for unknown filler
        if filler_type not in params:
            return {
                'shape': 'circle',
                'aspect_ratio': 1.0,
                'base_radius': 0.3,
                'color': 'green',
                'alpha': 0.5,
                'edgecolor': 'darkgreen',
                'linewidth': 1
            }
        
        return params[filler_type]
    
    def create_rve(self, filler_type, volume_fraction=0.15):
        """
        Create RVE for specific filler type
        
        Parameters:
        -----------
        filler_type : str
            Type of filler ('Carbon Fiber', 'Graphene Nanoplatelets', etc.)
        volume_fraction : float
            Target volume fraction (0-1)
        
        Returns:
        --------
        tuple : (fig, positions, orientations)
        """
        # Get particle parameters
        params = self.get_particle_params(filler_type)
        particle_shape = params['shape']
        aspect_ratio = params['aspect_ratio']
        base_radius = params['base_radius']
        color = params['color']
        alpha = params['alpha']
        
        # Calculate number of particles for target volume fraction
        if particle_shape == 'circle':
            particle_area = np.pi * base_radius**2
        elif particle_shape == 'ellipse':
            particle_area = np.pi * base_radius * (base_radius * aspect_ratio)
        elif particle_shape == 'rectangle':
            particle_area = (2*base_radius) * (2*base_radius * aspect_ratio)
        elif particle_shape == 'line':
            # Approximate area for line representation
            particle_area = base_radius * (base_radius * aspect_ratio) * 2
        else:
            particle_area = np.pi * base_radius**2
            
        rve_area = self.rve_size**2
        num_particles = int((volume_fraction * rve_area) / particle_area)
        num_particles = min(num_particles, 150)  # Limit for visualization clarity
        
        # Generate particle positions
        positions, orientations = self.random_sequential_adsorption(
            num_particles, base_radius, aspect_ratio
        )
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(0, self.rve_size)
        ax.set_ylim(0, self.rve_size)
        ax.set_aspect('equal')
        
        # Draw matrix background
        matrix_bg = Rectangle((0, 0), self.rve_size, self.rve_size, 
                            facecolor='lightgray', edgecolor='black', 
                            linewidth=2, alpha=0.3)
        ax.add_patch(matrix_bg)
        
        # Draw particles
        for i, (x, y) in enumerate(positions):
            if particle_shape == 'circle':
                circle = Circle((x, y), base_radius, 
                              facecolor=color, edgecolor='black', 
                              alpha=alpha, linewidth=1)
                ax.add_patch(circle)
            
            elif particle_shape == 'ellipse':
                width = 2 * base_radius * aspect_ratio
                height = 2 * base_radius
                ellipse = Ellipse((x, y), width, height, 
                                angle=np.degrees(orientations[i]),
                                facecolor=color, edgecolor='black',
                                alpha=alpha, linewidth=1)
                ax.add_patch(ellipse)
            
            elif particle_shape == 'rectangle':
                width = 2 * base_radius * aspect_ratio
                height = 2 * base_radius
                # Create rotated rectangle
                rect = Rectangle((x - width/2, y - height/2), width, height,
                               angle=np.degrees(orientations[i]),
                               facecolor=color, edgecolor='black',
                               alpha=alpha, linewidth=1)
                ax.add_patch(rect)
            
            elif particle_shape == 'line':
                # Represent CNTs as thin lines
                length = 2 * base_radius * aspect_ratio
                dx = length * np.cos(orientations[i]) / 2
                dy = length * np.sin(orientations[i]) / 2
                ax.plot([x - dx, x + dx], [y - dy, y + dy], 
                       color=color, linewidth=2, alpha=alpha)
        
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.set_title(f'{filler_type} in PEEK Matrix\nVolume Fraction: {volume_fraction:.1%}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add text with statistics
        stats_text = f'Particles: {len(positions)}\nVf: {volume_fraction:.1%}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return fig, positions, orientations
    
    def create_hybrid_rve(self, filler_types, volume_fractions, total_vf=0.15):
        """
        Create RVE with multiple filler types (hybrid microstructure)
        
        Parameters:
        -----------
        filler_types : list of str
            List of filler types to include
        volume_fractions : list of float
            Individual volume fractions for each filler (must sum to total_vf)
        total_vf : float
            Total volume fraction of all fillers
        
        Returns:
        --------
        matplotlib.figure.Figure : Hybrid microstructure figure
        """
        if len(filler_types) != len(volume_fractions):
            raise ValueError("Number of filler types must match number of volume fractions")
        
        if abs(sum(volume_fractions) - total_vf) > 1e-6:
            raise ValueError(f"Sum of volume fractions ({sum(volume_fractions)}) must equal total_vf ({total_vf})")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(0, self.rve_size)
        ax.set_ylim(0, self.rve_size)
        ax.set_aspect('equal')
        
        # Draw matrix background
        matrix_bg = Rectangle((0, 0), self.rve_size, self.rve_size, 
                            facecolor='lightgray', edgecolor='black', 
                            linewidth=2, alpha=0.3)
        ax.add_patch(matrix_bg)
        
        # Place fillers sequentially
        all_positions = []
        legend_elements = []
        
        for filler_type, vf in zip(filler_types, volume_fractions):
            params = self.get_particle_params(filler_type)
            
            # Calculate number of particles
            if params['shape'] == 'circle':
                particle_area = np.pi * params['base_radius']**2
            elif params['shape'] == 'ellipse':
                particle_area = np.pi * params['base_radius'] * (params['base_radius'] * params['aspect_ratio'])
            elif params['shape'] == 'rectangle':
                particle_area = (2*params['base_radius']) * (2*params['base_radius'] * params['aspect_ratio'])
            else:
                particle_area = np.pi * params['base_radius']**2
                
            rve_area = self.rve_size**2
            num_particles = int((vf * rve_area) / particle_area)
            num_particles = min(num_particles, 50)  # Limit per filler type
            
            # Generate positions avoiding overlap with previously placed particles
            positions, orientations = [], []
            for i in range(num_particles):
                attempts = 0
                placed = False
                
                while not placed and attempts < 5000:
                    x = np.random.uniform(params['base_radius']*2, self.rve_size - params['base_radius']*2)
                    y = np.random.uniform(params['base_radius']*2, self.rve_size - params['base_radius']*2)
                    
                    # Check overlap with all previously placed particles
                    overlap = False
                    for pos in all_positions:
                        dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                        if dist < params['base_radius'] * 3:  # Conservative overlap check
                            overlap = True
                            break
                    
                    if not overlap:
                        positions.append((x, y))
                        orientations.append(np.random.uniform(0, np.pi))
                        all_positions.append((x, y))
                        placed = True
                    
                    attempts += 1
            
            # Draw particles for this filler type
            for i, (x, y) in enumerate(positions):
                if params['shape'] == 'ellipse':
                    width = 2 * params['base_radius'] * params['aspect_ratio']
                    height = 2 * params['base_radius']
                    ellipse = Ellipse((x, y), width, height,
                                    angle=np.degrees(orientations[i]),
                                    facecolor=params['color'], 
                                    edgecolor='black',
                                    alpha=params['alpha'], linewidth=1)
                    ax.add_patch(ellipse)
                elif params['shape'] == 'rectangle':
                    width = 2 * params['base_radius'] * params['aspect_ratio']
                    height = 2 * params['base_radius']
                    rect = Rectangle((x - width/2, y - height/2), width, height,
                                   angle=np.degrees(orientations[i]),
                                   facecolor=params['color'], edgecolor='black',
                                   alpha=params['alpha'], linewidth=1)
                    ax.add_patch(rect)
                else:
                    circle = Circle((x, y), params['base_radius'],
                                  facecolor=params['color'], edgecolor='black',
                                  alpha=params['alpha'], linewidth=1)
                    ax.add_patch(circle)
            
            # Add to legend
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=params['color'],
                                             markersize=10, label=filler_type))
        
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        
        # Create combination name for title
        combo_name = '+'.join(filler_types)
        ax.set_title(f'Hybrid Composite: {combo_name} in PEEK\nTotal Vf: {total_vf:.1%}', 
                    fontsize=14, fontweight='bold')
        
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_rve_comparison(self, filler_properties):
        """
        Create a comparison figure with RVEs for all filler types
        
        Parameters:
        -----------
        filler_properties : dict
            Properties of all fillers
        
        Returns:
        --------
        matplotlib.figure.Figure : Comparison figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()
        
        for idx, (filler_name, props) in enumerate(filler_properties.items()):
            ax = axes[idx]
            
            # Generate RVE for this filler
            rve_fig, positions, orientations = self.create_rve(filler_name, volume_fraction=0.15)
            
            # Copy the plot to the comparison figure
            ax.set_xlim(0, self.rve_size)
            ax.set_ylim(0, self.rve_size)
            ax.set_aspect('equal')
            
            # Draw matrix
            matrix_bg = Rectangle((0, 0), self.rve_size, self.rve_size,
                                facecolor='lightgray', edgecolor='black',
                                linewidth=2, alpha=0.3)
            ax.add_patch(matrix_bg)
            
            # Get filler parameters
            params = self.get_particle_params(filler_name)
            
            # Draw particles (simplified for comparison)
            for i, (x, y) in enumerate(positions[:30]):  # Limit for clarity
                if params['shape'] == 'ellipse':
                    width = 2 * params['base_radius'] * params['aspect_ratio']
                    height = 2 * params['base_radius']
                    ellipse = Ellipse((x, y), width/2, height/2,  # Smaller for comparison
                                    angle=np.degrees(orientations[i]) if i < len(orientations) else 0,
                                    facecolor=params['color'], edgecolor='black',
                                    alpha=0.7, linewidth=1)
                    ax.add_patch(ellipse)
                else:
                    circle = Circle((x, y), 0.25, facecolor=params['color'], 
                                  edgecolor='black', alpha=0.7)
                    ax.add_patch(circle)
            
            ax.set_title(f'{filler_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('PEEK Composite Microstructures (15% Filler Volume Fraction)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def create_multi_filler_comparison(self, filler_properties):
        """
        Create a comprehensive figure showing single and hybrid microstructures
        
        Returns:
        --------
        matplotlib.figure.Figure : Multi-panel figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Single filler microstructures (top row)
        for i, filler in enumerate(['Carbon Fiber', 'Graphene Nanoplatelets', 
                                     'Carbon Nanotubes', 'Glass Fiber']):
            ax = fig.add_subplot(3, 4, i+1)
            ax.set_xlim(0, self.rve_size)
            ax.set_ylim(0, self.rve_size)
            ax.set_aspect('equal')
            ax.set_title(filler.split()[0], fontsize=10)
            
            # Simple representation
            params = self.get_particle_params(filler)
            for j in range(10):
                x, y = np.random.uniform(1, 9, 2)
                if params['shape'] == 'ellipse':
                    ellipse = Ellipse((x, y), 0.8, 0.3, 
                                    angle=np.random.uniform(0, 180),
                                    facecolor=params['color'], alpha=0.7)
                    ax.add_patch(ellipse)
                else:
                    circle = Circle((x, y), 0.3, facecolor=params['color'], alpha=0.7)
                    ax.add_patch(circle)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hybrid microstructures (bottom rows)
        hybrids = [
            (['Carbon Fiber', 'Graphene Nanoplatelets'], [0.075, 0.075]),
            (['Carbon Fiber', 'Carbon Nanotubes'], [0.075, 0.075]),
            (['Graphene Nanoplatelets', 'Carbon Nanotubes'], [0.075, 0.075]),
            (['Carbon Fiber', 'Graphene Nanoplatelets', 'Carbon Nanotubes'], [0.05, 0.05, 0.05])
        ]
        
        for i, (fillers, vfs) in enumerate(hybrids):
            ax = fig.add_subplot(3, 4, 5+i)
            ax.set_xlim(0, self.rve_size)
            ax.set_ylim(0, self.rve_size)
            ax.set_aspect('equal')
            ax.set_title('+'.join([f.split()[0] for f in fillers]), fontsize=9)
            
            for filler, vf in zip(fillers, vfs):
                params = self.get_particle_params(filler)
                num = int(vf * 100)
                for j in range(min(num, 5)):
                    x, y = np.random.uniform(1, 9, 2)
                    if params['shape'] == 'ellipse':
                        ellipse = Ellipse((x, y), 0.8, 0.3,
                                        angle=np.random.uniform(0, 180),
                                        facecolor=params['color'], alpha=0.7)
                        ax.add_patch(ellipse)
                    else:
                        circle = Circle((x, y), 0.3, facecolor=params['color'], alpha=0.7)
                        ax.add_patch(circle)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('PEEK Composite Microstructures: Single and Hybrid Systems', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig