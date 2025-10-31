import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import matplotlib as mpl
from scipy import stats
warnings.filterwarnings('ignore')

# 设置全局字体为衬线字体 - 适合论文发表
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用 STIX 字体


class PerBinVisualizer:
    """Generate publication-quality visualization for each bin"""
    
    def __init__(self, dict_path: str, save_dir: str = "./bin_visualizations"):
        self.dict_path = Path(dict_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        
        # Color schemes - Keep history/future unchanged, optimize others
        self.colors = {
            'dense': '#2ECC71',    # Green - more vivid
            'noise': '#F39C12',    # Orange - more vivid
            'history': '#FF6B6B',  # Keep original red
            'future': '#4ECDC4',   # Keep original teal
        }
        
        # Feature names for manual features (5D)
        self.feature_names = [
            'Speed',
            'Acceleration', 
            'Heading Rate',
            'Lateral Vel.',
            'Longitudinal Acc.'
        ]
    
    def load_dictionary(self) -> bool:
        """Load dictionary from pt file"""
        try:
            print(f"Loading dictionary from {self.dict_path}...")
            self.data = torch.load(self.dict_path, map_location='cpu')
            print(f"✓ Dictionary loaded successfully")
            
            # Count bins
            n_bins = len(self.data['dictionary'])
            print(f"Found {n_bins} bins to visualize")
            return True
        except Exception as e:
            print(f"✗ Error loading dictionary: {e}")
            return False
    
    def visualize_single_bin(self, bin_name: str, bin_data: Dict, 
                           save: bool = True, show: bool = False):
        """Create publication-quality visualization for a single bin"""
        
        # Extract bin type
        bin_type = bin_name.split('_')[0]
        bin_color = self.colors.get(bin_type, '#666666')
        
        # Convert data to numpy if needed
        histories = bin_data['fps_histories']  # [N, 20, 5]
        futures = bin_data['fps_futures']      # [N, 80, 3]
        features = bin_data['features']        # [N, 5]
        deep_features = bin_data['deep_features']  # [N, 256]
        
        if isinstance(histories, torch.Tensor):
            histories = histories.numpy()
        if isinstance(futures, torch.Tensor):
            futures = futures.numpy()
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(deep_features, torch.Tensor):
            deep_features = deep_features.numpy()
        
        n_samples = len(histories)
        
        # Sample trajectories if too many (for clearer visualization)
        max_traj_display = 50
        if n_samples > max_traj_display:
            sample_indices = np.random.choice(n_samples, max_traj_display, replace=False)
        else:
            sample_indices = np.arange(n_samples)
        
        # Create figure with optimized layout for papers (taller, narrower)
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35, 
                     height_ratios=[1.2, 1], width_ratios=[1, 1])
        
        # ========== Main Trajectory Plot (Top, spanning 2 columns) ==========
        ax_main = fig.add_subplot(gs[0, :])
        
        # Plot sampled trajectories with better visibility
        for i in sample_indices:
            hist = histories[i]  # [20, 5]
            fut = futures[i]     # [80, 3]
            
            # History trajectories (x, y from first 2 dimensions)
            ax_main.plot(hist[:, 0], hist[:, 1], 
                        color=self.colors['history'], alpha=0.5, 
                        linewidth=1.5, zorder=1)
            
            # Future trajectories
            ax_main.plot(fut[:, 0], fut[:, 1], 
                        color=self.colors['future'], alpha=0.5, 
                        linewidth=2, zorder=2)
        
        # Mark transition point for a few representative trajectories
        for i in sample_indices[:5]:
            hist = histories[i]
            ax_main.scatter(hist[-1, 0], hist[-1, 1], 
                          c='black', s=40, marker='o', zorder=3, 
                          edgecolors='white', linewidths=0.5)
        
        # Add legend with larger font
        history_line = plt.Line2D([0], [0], color=self.colors['history'], 
                                 linewidth=2.5, label='History (2s, 20 pts)')
        future_line = plt.Line2D([0], [0], color=self.colors['future'], 
                                linewidth=2.5, label='Future (8s, 80 pts)')
        ax_main.legend(handles=[history_line, future_line], loc='upper right', 
                      fontsize=11, framealpha=0.95, edgecolor='gray')
        
        ax_main.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax_main.tick_params(labelsize=10)
        ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax_main.axis('equal')
        
        # Add statistics text box with better styling
        traj_lengths = []
        for i in range(n_samples):
            fut = futures[i]
            diffs = np.diff(fut[:, :2], axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
            traj_lengths.append(length)
        
        stats_text = f"Type: {bin_type.upper()}\n"
        stats_text += f"N = {n_samples}\n"
        stats_text += f"Avg. Length: {np.mean(traj_lengths):.1f}m ± {np.std(traj_lengths):.1f}m"
        
        ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', 
                             alpha=0.9, edgecolor='gray', linewidth=1.5))
        
        # Add subplot label
        ax_main.text(0.02, 1.05, '(a)', transform=ax_main.transAxes,
                    fontsize=14, fontweight='bold')
        
        # ========== Manual Features Statistics ==========
        ax_manual = fig.add_subplot(gs[1, 0])
        
        # Calculate statistics for manual features
        feature_stats_data = []
        for i in range(features.shape[1]):
            feat_vals = features[:, i]
            feature_stats_data.append({
                'name': self.feature_names[i],
                'mean': np.mean(feat_vals),
                'std': np.std(feat_vals),
            })
        
        # Create bar plot with better styling
        x_pos = np.arange(len(self.feature_names))
        means = [f['mean'] for f in feature_stats_data]
        stds = [f['std'] for f in feature_stats_data]
        
        bars = ax_manual.bar(x_pos, means, yerr=stds, capsize=4,
                            color=bin_color, alpha=0.75, edgecolor='black', 
                            linewidth=1.2, error_kw={'linewidth': 1.5})
        
        ax_manual.set_xticks(x_pos)
        ax_manual.set_xticklabels([name for name in self.feature_names], 
                                  fontsize=9, rotation=15, ha='right')
        ax_manual.set_ylabel('Feature Value', fontsize=11, fontweight='bold')
        ax_manual.set_title('Manual Features (5D)', fontsize=12, fontweight='bold', pad=10)
        ax_manual.tick_params(labelsize=10)
        ax_manual.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        
        # Add subplot label
        ax_manual.text(0.02, 1.08, '(b)', transform=ax_manual.transAxes,
                      fontsize=14, fontweight='bold')
        
        # ========== Deep Feature Statistics ==========
        ax_deep = fig.add_subplot(gs[1, 1])
        
        # Calculate deep feature statistics
        deep_norms = np.linalg.norm(deep_features, axis=1)
        
        # Plot histogram with better styling
        n, bins, patches_hist = ax_deep.hist(deep_norms, bins=25, color=bin_color, 
                                            alpha=0.75, edgecolor='black', 
                                            density=True, linewidth=1.2)
        
        # Add normal distribution fit with thicker line
        mu, std = stats.norm.fit(deep_norms)
        xmin, xmax = ax_deep.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax_deep.plot(x, p, 'k-', linewidth=2.5, 
                    label=f'Fit: μ={mu:.1f}, σ={std:.1f}')
        
        ax_deep.set_xlabel('L2 Norm', fontsize=11, fontweight='bold')
        ax_deep.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax_deep.set_title('Deep Features (256D)', fontsize=12, fontweight='bold', pad=10)
        ax_deep.tick_params(labelsize=10)
        ax_deep.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax_deep.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray')
        
        # Add subplot label
        ax_deep.text(0.02, 1.08, '(c)', transform=ax_deep.transAxes,
                    fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{bin_name}_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"  ✓ Saved {bin_name} visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_all_bins(self, max_bins: Optional[int] = None, 
                          show: bool = False):
        """Generate visualizations for all bins in the dictionary"""
        
        if self.data is None:
            print("Please load dictionary first")
            return
        
        dictionary = self.data['dictionary']
        bin_names = sorted(list(dictionary.keys()))
        
        if max_bins:
            bin_names = bin_names[:max_bins]
        
        print(f"\nGenerating publication-quality visualizations for {len(bin_names)} bins...")
        print("="*60)
        
        # Group bins by type for summary
        bin_types_count = {'dense': 0, 'noise': 0, 'other': 0}
        
        for idx, bin_name in enumerate(bin_names, 1):
            print(f"[{idx}/{len(bin_names)}] Processing {bin_name}...")
            
            bin_data = dictionary[bin_name]
            bin_type = bin_name.split('_')[0]
            
            if bin_type in bin_types_count:
                bin_types_count[bin_type] += 1
            else:
                bin_types_count['other'] += 1
            
            # Generate visualization
            self.visualize_single_bin(bin_name, bin_data, save=True, show=show)
        
        print("\n" + "="*60)
        print("Visualization Summary:")
        for bin_type, count in bin_types_count.items():
            if count > 0:
                print(f"  - {bin_type.capitalize()} bins: {count}")
        print(f"Total visualizations created: {len(bin_names)}")
        print(f"Saved to: {self.save_dir}")


def main():
    """Main execution"""
    # Configuration
    dict_path = "/extra_disk/yx/paper2/history/trajectory_cache/diversity_aware_dictionary_final.pt"
    save_dir = "/extra_disk/yx/paper2/history/diversity_aware_dictionary_final2_optimized"
    
    # Initialize visualizer
    visualizer = PerBinVisualizer(dict_path, save_dir)
    
    # Load dictionary
    if not visualizer.load_dictionary():
        return
    
    print("\n" + "="*60)
    print("PUBLICATION-QUALITY PER-BIN VISUALIZATION")
    print("="*60)
    
    # Visualize all bins
    visualizer.visualize_all_bins(max_bins=None, show=False)
    
    # Or visualize only first N bins for testing
    # visualizer.visualize_all_bins(max_bins=5, show=False)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"All visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
