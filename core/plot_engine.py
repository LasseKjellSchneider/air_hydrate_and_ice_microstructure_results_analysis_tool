import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

colors_lava = ['#000080', '#800080', '#ff8c00', '#ffff00']
lava_cmap = LinearSegmentedColormap.from_list('lava', colors_lava)

colors_div = ["#c834db", "#000a97", "#3ce7d9"] 
div_cmap = LinearSegmentedColormap.from_list('div', colors_div)

def get_cmap(cmap_name):
    if cmap_name == 'lava': return lava_cmap
    if cmap_name == 'div': return div_cmap
    try: return plt.get_cmap(cmap_name)
    except: return lava_cmap

def get_color(color_name):
    mapping = {
        'Blue': '#3498db', 
        'Red': '#e74c3c', 
        'Green': '#2ecc71', 
        'Purple': '#9b59b6', 
        'Orange': '#f39c12', 
        'Cyan': '#00ffff',
        'Teal (#298c8c)': '#298c8c',
        'Dark Purple (#721085)': '#721085',
        'Warm Orange (#f1a226)': '#f1a226'
    }
    return mapping.get(color_name, '#3498db')

class PlotEngine:
    @staticmethod
    def _apply_theme(fig, ax, is_preview, dark_mode, show_grid=False, label_size=10):
        bg_color = '#2b2b2b' if dark_mode else 'white'
        fg_color = 'white' if dark_mode else 'black'
        spine_color = 'gray' if dark_mode else 'black'
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        actual_label_size = 6 if is_preview else label_size
        ax.tick_params(colors=fg_color, labelsize=actual_label_size)
        
        for spine in ax.spines.values(): 
            spine.set_color(spine_color)
            
        if not is_preview:
            ax.xaxis.label.set_color(fg_color)
            ax.yaxis.label.set_color(fg_color)
            ax.xaxis.label.set_fontsize(actual_label_size + 2)
            ax.yaxis.label.set_fontsize(actual_label_size + 2)
            ax.title.set_color(fg_color)
            ax.title.set_fontsize(actual_label_size + 4)
            
        if show_grid and not is_preview:
            ax.grid(True, linestyle='--', alpha=0.3, color=fg_color)

    @staticmethod
    def _apply_limits(ax, x_min, x_max, y_min, y_max):
        if x_min is not None: ax.set_xlim(left=x_min)
        if x_max is not None: ax.set_xlim(right=x_max)
        if y_min is not None: ax.set_ylim(bottom=y_min)
        if y_max is not None: ax.set_ylim(top=y_max)

    @staticmethod
    def _generate_2d_grids(df, sample_height_um, tile_mult, blur_val):
        base_size = 500.0
        tile_size = base_size * tile_mult
        
        # Normalize coordinates so the origin is at (0, 0).
        x_offset = df['x_um'].min()
        y_offset = df['y_um'].min()
        
        xs = df['x_um'].values - x_offset
        ys = df['y_um'].values - y_offset
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        x_edges = np.arange(np.floor(x_min/tile_size)*tile_size, np.ceil(x_max/tile_size)*tile_size + tile_size, tile_size)
        y_edges = np.arange(np.floor(y_min/tile_size)*tile_size, np.ceil(y_max/tile_size)*tile_size + tile_size, tile_size)
        x_centers, y_centers = x_edges[:-1] + tile_size / 2, y_edges[:-1] + tile_size / 2
        
        density = np.zeros((len(y_centers), len(x_centers)))
        area = np.full((len(y_centers), len(x_centers)), np.nan)
        aspect = np.full((len(y_centers), len(x_centers)), np.nan)
        
        window_size = tile_size * blur_val
        window_vol_mm3 = (window_size / 1000.0)**2 * (sample_height_um / 1000.0)
        
        areas = df['area_um2'].values
        aspects = df['aspect_ratio'].values
        
        for i, yc in enumerate(y_centers):
            for j, xc in enumerate(x_centers):
                mask = (xs >= xc - window_size/2) & (xs < xc + window_size/2) & (ys >= yc - window_size/2) & (ys < yc + window_size/2)
                c = np.sum(mask)
                density[i, j] = c / window_vol_mm3 if window_vol_mm3 > 0 else 0
                if c > 0:
                    area[i, j] = np.median(areas[mask])
                    aspect[i, j] = np.median(aspects[mask])
                    
        return x_edges.copy(), y_edges.copy(), density.copy(), area.copy(), aspect.copy()

    @staticmethod
    def _calc_image_grid(positions):
        if not positions: return [], []
        x_vals = sorted(list(set(round(p['positionX'], 1) for p in positions)))
        y_vals = sorted(list(set(round(p['positionY'], 1) for p in positions)))
        
        dx = [x_vals[i+1] - x_vals[i] for i in range(len(x_vals)-1)]
        step_x = np.median(dx) if dx else 1400.0
        width_um = step_x / 0.75
        
        dy = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
        step_y = np.median(dy) if dy else 1400.0
        height_um = step_y / 0.75
        
        centers_x = [x + width_um / 2 for x in x_vals]
        centers_y = [y + height_um / 2 for y in y_vals]
        
        def get_edges(centers, img_size):
            if len(centers) < 2: return [centers[0]-img_size/2, centers[0]+img_size/2] if centers else []
            edges = [centers[0] - (centers[1]-centers[0])/2]
            for i in range(len(centers)-1):
                edges.append((centers[i] + centers[i+1])/2)
            edges.append(centers[-1] + (centers[-1]-centers[-2])/2)
            return edges
            
        return get_edges(centers_x, width_um), get_edges(centers_y, height_um)

    # ================= BOX A: Distributions & Scatter =================
    @staticmethod
    def create_dist(df, col, bins=50, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        
        color = get_color(base_color)
        counts, bin_edges, _ = ax.hist(df[col], bins=bins, color=color, edgecolor='black', alpha=0.8)
        
        title = "Area" if col == 'area_um2' else "Aspect Ratio"
        if not is_preview:
            if show_title: ax.set_title(f"Distribution of {title}")
            ax.set_xlabel(f"{title} {'[µm²]' if col=='area_um2' else ''}")
            ax.set_ylabel("Count")
            PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
        else:
            ax.set_title(f"{title} Dist.", color="white" if dark_mode else "black", fontsize=9)
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    @staticmethod
    def create_scatter_corr(df, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        
        color = get_color(base_color)
        ax.scatter(df['area_um2'], df['aspect_ratio'], alpha=0.3, color=color, edgecolor='none', s=10 if is_preview else 20)
        
        if not is_preview and not df.empty:
            norm_x = (df['area_um2'] - df['area_um2'].min()) / (df['area_um2'].max() - df['area_um2'].min() + 1e-9)
            norm_y = (df['aspect_ratio'] - df['aspect_ratio'].min()) / (df['aspect_ratio'].max() - df['aspect_ratio'].min() + 1e-9)
            
            grid_x, grid_y = (norm_x * 9.99).astype(int), (norm_y * 9.99).astype(int)
            df_temp = df.copy()
            df_temp['grid_id'] = grid_x.astype(str) + "_" + grid_y.astype(str)
            
            sampled = df_temp.groupby('grid_id').sample(1)
            if len(sampled) < 100:
                remaining = df_temp[~df_temp.index.isin(sampled.index)]
                needed = min(100 - len(sampled), len(remaining))
                if needed > 0: sampled = pd.concat([sampled, remaining.sample(needed)])
            
            fig.clickable_points = sampled
            ax.scatter(sampled['area_um2'], sampled['aspect_ratio'], color='#00ffff', edgecolor='black', linewidth=1, s=40, zorder=5)
            
            hover_hl = ax.scatter([], [], facecolors='none', edgecolors='#ff00ff', s=120, linewidth=2, zorder=10)
            fig.hover_highlight = hover_hl
            
            if show_title: ax.set_title("Area vs. Aspect Ratio (100 Clickable Samples)")
            ax.set_xlabel("Area [µm²]"); ax.set_ylabel("Aspect Ratio")
            PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
        else:
            ax.set_title("Area vs Aspect", color="white" if dark_mode else "black", fontsize=9)
            
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    @staticmethod
    def create_scatter_tiles(df, sample_height_um, glob_dens, glob_area, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        
        if df.empty: return fig
        x_edges, y_edges, dens, area, _ = PlotEngine._generate_2d_grids(df, sample_height_um, 1, 3)
        x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
        y_centers = y_edges[:-1] + (y_edges[1] - y_edges[0]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        d_flat, a_flat, x_flat, y_flat = dens.flatten(), area.flatten(), X.flatten(), Y.flatten()
        m = ~np.isnan(d_flat) & ~np.isnan(a_flat) & (d_flat > 0)
        
        color = get_color(base_color)
        ax.scatter(d_flat[m], a_flat[m], alpha=0.7, color=color, edgecolor='black', linewidth=0.5, s=10 if is_preview else 30)
        
        if not is_preview:
            # Restore the offsets so click handling can match the original image coordinates.
            x_offset, y_offset = df['x_um'].min(), df['y_um'].min()
            df_tiles = pd.DataFrame({'density': d_flat[m], 'area': a_flat[m], 'x_um': x_flat[m] + x_offset, 'y_um': y_flat[m] + y_offset})
            fig.tile_points = df_tiles
            
            hover_hl = ax.scatter([], [], facecolors='none', edgecolors='#00ffff', s=120, linewidth=2, zorder=10)
            fig.hover_highlight = hover_hl
            
            fg = 'white' if dark_mode else 'black'
            ax.axvline(glob_dens, color=fg, linestyle='--', alpha=0.5)
            ax.axhline(glob_area, color=fg, linestyle=':', alpha=0.5)
            
            if show_title: ax.set_title("Tile Analysis: Area vs. Density (Click to open Stack)")
            ax.set_xlabel("Local Density [hyd/mm³]"); ax.set_ylabel("Local Median Area [µm²]")
            PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
        else:
            ax.set_title("Tile Scatter", color="white" if dark_mode else "black", fontsize=9)
            
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    @staticmethod
    def create_tile_heatmap(df, sample_height_um, dark_mode=True, show_title=True, show_grid=False, label_size=10, cmap_name='lava', base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        
        if df.empty: return fig
        _, _, dens, area, _ = PlotEngine._generate_2d_grids(df, sample_height_um, 1, 3)
        d_flat, a_flat = dens.flatten(), area.flatten()
        m = ~np.isnan(d_flat) & ~np.isnan(a_flat) & (d_flat > 0)
        d_val, a_val = d_flat[m], a_flat[m]
        if len(d_val) == 0: return fig

        window_vol_mm3 = (1500.0 / 1000.0)**2 * (sample_height_um / 1000.0)
        step_d = 1.0 / window_vol_mm3
        max_c = np.ceil(np.max(d_val) / step_d)
        x_edges = np.arange(0.5 * step_d, (max_c + 1.5) * step_d, step_d)
        
        step_a = 2000.0
        max_a = np.max(a_val)
        y_edges = np.arange(0, max_a + step_a * 2, step_a)
        
        H, xedges, yedges = np.histogram2d(d_val, a_val, bins=[x_edges, y_edges])
        H_masked = np.ma.masked_where(H == 0, H) 
        
        cmap = get_cmap(cmap_name)
        mesh = ax.pcolormesh(xedges, yedges, H_masked.T, cmap=cmap)

        if not is_preview:
            if show_title: ax.set_title(f"Tile Heatmap (X-Bins: 1 Hyd/Tile, Y-Bins: {int(step_a)} [µm²])")
            ax.set_xlabel("Local Density [hyd/mm³]"); ax.set_ylabel("Local Median Area [µm²]")
            PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
            
            cb = fig.colorbar(mesh, ax=ax)
            fg = 'white' if dark_mode else 'black'
            cb.set_label("Number of Tiles", color=fg, fontsize=label_size)
            cb.ax.yaxis.set_tick_params(color=fg, labelcolor=fg, labelsize=label_size)
        else:
            ax.set_title("Tile Heatmap", color="white" if dark_mode else "black", fontsize=9)
            
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    # ================= BOX B: Heatmaps =================
    @staticmethod
    def create_heatmap(df, sh_um, g_dens, g_area, map_type, tile_mult=1, blur_val=3, dark_mode=True, show_title=True, show_grid=False, label_size=10, cmap_name='lava', base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(6, 5), dpi=100, is_preview=False, positions=None, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        if df.empty: return fig
            
        x_edges, y_edges, dens, area, aspect = PlotEngine._generate_2d_grids(df, sh_um, tile_mult, blur_val)
        
        x_offset = df['x_um'].min()
        y_offset = df['y_um'].min()
        
        if map_type == "Density Heatmap":
            grid, label, title = dens, "Density [hyd/mm³]", "Density Map"
            current_cmap = get_cmap(cmap_name)
        elif map_type == "Area Heatmap":
            grid, label, title = area, "Median Area [µm²]", "Area Map"
            current_cmap = get_cmap(cmap_name)
        elif map_type == "Aspect Heatmap":
            grid, label, title = aspect, "Median Aspect", "Aspect Ratio Map"
            current_cmap = get_cmap(cmap_name)
        elif map_type == "Relation Heatmap":
            with np.errstate(divide='ignore', invalid='ignore'):
                R = (area / g_area) * (g_dens / dens)
                grid = np.full_like(R, np.nan)
                mask_pos = (R > 1) & ~np.isnan(R) & ~np.isinf(R)
                grid[mask_pos] = R[mask_pos] - 1.0
                mask_neg = (R <= 1) & (R > 0) & ~np.isnan(R) & ~np.isinf(R)
                grid[mask_neg] = -(1.0 / R[mask_neg]) + 1.0
                
            label, title = "Relation Score", "Area-Density Relation"
            current_cmap = get_cmap(cmap_name)
            if cmap_name == 'lava': current_cmap = div_cmap 

        if map_type == "Relation Heatmap":
            valid_vals = grid[~np.isnan(grid) & ~np.isinf(grid)]
            if len(valid_vals) > 0:
                max_abs = np.max(np.abs(valid_vals))
                if max_abs == 0: max_abs = 1.0
                vmin, vmax = -max_abs, max_abs
            else: vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = np.nanmin(grid), np.nanmax(grid)

        mesh = ax.pcolormesh(x_edges, y_edges, grid, cmap=current_cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal', adjustable='box')
        
        # Format the axis tick labels in millimeters.
        mm_formatter = ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}")
        ax.xaxis.set_major_formatter(mm_formatter)
        ax.yaxis.set_major_formatter(mm_formatter)
        
        # Determine the coordinate limits before applying the axis orientation.
        x_dmin, x_dmax = x_edges.min(), x_edges.max()
        y_dmin, y_dmax = y_edges.min(), y_edges.max()
        
        if not is_preview:
            if show_title: ax.set_title(f"{title} [Tile: {int(500*tile_mult)}µm, Blur: {blur_val}x]")
            ax.set_xlabel("X [mm] -> top")
            ax.set_ylabel("Y [mm]")
            
            # Convert user-specified limits from millimeters to micrometers.
            if x_min is not None: x_dmin = x_min * 1000.0
            if x_max is not None: x_dmax = x_max * 1000.0
            if y_min is not None: y_dmin = y_min * 1000.0
            if y_max is not None: y_dmax = y_max * 1000.0
            
            cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            fg = 'white' if dark_mode else 'black'
            cb.set_label(label, color=fg, fontsize=label_size)
            cb.ax.yaxis.set_tick_params(color=fg, labelcolor=fg, labelsize=label_size)
            
            if map_type == "Relation Heatmap":
                cb.ax.axhline(0, color=fg, linewidth=1, linestyle='--')
            else:
                m_val = np.nanmean(grid)
                if not np.isnan(m_val): cb.ax.axhline(m_val, color=fg, linewidth=2, linestyle='--')
                
            if positions:
                shifted_positions = []
                for p in positions:
                    p_new = p.copy()
                    p_new['positionX'] -= x_offset
                    p_new['positionY'] -= y_offset
                    shifted_positions.append(p_new)
                    
                x_img_edges, y_img_edges = PlotEngine._calc_image_grid(shifted_positions)
                
                grid_lines = []
                for xe in x_img_edges:
                    grid_lines.append(ax.axvline(xe, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, visible=False, zorder=5))
                for ye in y_img_edges:
                    grid_lines.append(ax.axhline(ye, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, visible=False, zorder=5))
                
                hover_rect = Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#e74c3c', linewidth=2, alpha=0.25, visible=False, zorder=10)
                ax.add_patch(hover_rect)
                
                fig.image_x_edges = x_img_edges
                fig.image_y_edges = y_img_edges
                fig.grid_lines = grid_lines
                fig.hover_rect = hover_rect
                
                # Store the offsets on the figure so the popup can map clicks to TIFF coordinates.
                fig.offset_x = x_offset
                fig.offset_y = y_offset
        else:
            ax.set_title(title, color="white" if dark_mode else "black", fontsize=9)
            
        # Reverse the Y axis so the minimum coordinate is displayed at the top.
        ax.set_xlim(left=x_dmin, right=x_dmax)
        ax.set_ylim(bottom=y_dmax, top=y_dmin)
            
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    # ================= BOX C: Orientation & Profiles =================
    @staticmethod
    def create_rose(df, is_half=False, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Red', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(3, 3), dpi=100, is_preview=False, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        
        color = get_color(base_color)
        if not is_half:
            angles = np.deg2rad(df['angle_deg']); angles = np.concatenate([angles, angles + np.pi])
            counts, bins = np.histogram(angles, bins=np.linspace(0, 2*np.pi, 37))
            ax.bar(bins[:-1], counts, width=2*np.pi/36, bottom=0.0, color=color, edgecolor='black', alpha=0.7)
            title = "Rose Diagram"
        else:
            if base_color == 'Red': color = '#2980b9' 
            ax.scatter(np.deg2rad(df['angle_deg'] % 180), df['aspect_ratio'], alpha=0.6, color=color, s=10 if is_preview else 30)
            ax.set_thetamin(0); ax.set_thetamax(180)
            ax.set_rorigin(1.0)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            title = "Half-Rose"
            
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)      
        
        if is_half:
            ax.set_xticks(np.deg2rad([0, 90, 180]))
            ax.set_xticklabels(['-y\n[0°]', '+x\n[90°]', '+y\n[180°]'])
        else:
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels(['-y\n[0°]', '+x\n[90°]', '+y\n[180°]', '-x\n[270°]'])

        # Scale tick-label padding with the selected label size.
        pad_val = 5 if is_preview else max(10, label_size + 5)
        ax.tick_params(axis='x', pad=pad_val)

        if not is_preview:
            if show_title: ax.set_title(title, pad=25 + pad_val)
            if y_min is not None: ax.set_rorigin(y_min)
            if y_max is not None: ax.set_rmax(y_max)
        else:
            ax.set_title(title, color="white" if dark_mode else "black", fontsize=9, pad=5)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    @staticmethod
    def create_half_rose_heatmap(df, angle_bin_deg=5, aspect_bin_num=5, dark_mode=True, show_title=True, show_grid=False, label_size=10, cmap_name='lava', base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(3, 3), dpi=100, is_preview=False, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        
        if df.empty: return fig

        angles = df['angle_deg'] % 180
        aspects = df['aspect_ratio']
        aspect_bin = aspect_bin_num / 30.0

        theta_edges = np.deg2rad(np.arange(0, 180 + angle_bin_deg, angle_bin_deg))
        r_max = np.max(aspects)
        r_edges = np.arange(1.0, r_max + aspect_bin * 2, aspect_bin)

        H, t_edges, r_edges = np.histogram2d(np.deg2rad(angles), aspects, bins=[theta_edges, r_edges])
        H_masked = np.ma.masked_where(H == 0, H)

        T, R = np.meshgrid(t_edges, r_edges)
        cmap = get_cmap(cmap_name)
        mesh = ax.pcolormesh(T, R, H_masked.T, cmap=cmap)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rorigin(1.0)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.deg2rad([0, 90, 180]))
        ax.set_xticklabels(['-y\n[0°]', '+x\n[90°]', '+y\n[180°]'])

        # Adjust tick-label padding to accommodate the selected label size.
        pad_val = 5 if is_preview else max(10, label_size + 5)
        ax.tick_params(axis='x', pad=pad_val)

        if not is_preview:
            if show_title: ax.set_title(f"Half-Rose Heatmap [ΔAngle={angle_bin_deg}°, ΔAspect={aspect_bin:.2f}]", pad=25 + pad_val)
            if y_min is not None: ax.set_rorigin(y_min)
            if y_max is not None: ax.set_rmax(y_max)
            
            cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.1)
            fg = 'white' if dark_mode else 'black'
            cb.set_label("Count", color=fg, fontsize=label_size)
            cb.ax.yaxis.set_tick_params(color=fg, labelcolor=fg, labelsize=label_size)
        else:
            ax.set_title("Half-Rose Heatmap", color="white" if dark_mode else "black", fontsize=9, pad=5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    @staticmethod
    def create_profile(df, axis_col, metric, bin_width=100.0, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(4, 3), dpi=100, is_preview=False, **kwargs):
        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        PlotEngine._apply_theme(fig, ax, is_preview, dark_mode, show_grid, label_size)
        if df.empty: return fig
            
        # Normalize the profile coordinates so they start at zero.
        offset = df[axis_col].min()
        df_copy = df.copy()
        df_copy[axis_col] = df_copy[axis_col] - offset
            
        min_v, max_v = np.floor(df_copy[axis_col].min()/bin_width)*bin_width, np.ceil(df_copy[axis_col].max()/bin_width)*bin_width
        bins = np.arange(min_v, max_v + bin_width, bin_width)
        
        df_copy['bin'] = pd.cut(df_copy[axis_col], bins=bins)
        profile = df_copy.groupby('bin', observed=False).agg(count=(axis_col, 'size'), area=('area_um2', 'median'), aspect=('aspect_ratio', 'median')).reset_index()
        profile['bin_center'] = profile['bin'].apply(lambda x: x.mid).astype(float)
        
        color = get_color(base_color)
        if base_color == 'Blue' and metric == 'count': color = '#34495e'
        elif base_color == 'Blue' and metric == 'area': color = '#e74c3c'
        elif base_color == 'Blue' and metric == 'aspect': color = '#f39c12'
        
        if metric == 'count':
            ax.bar(profile['bin_center'], profile['count'], width=bin_width*0.9, color=color, edgecolor='black')
            if not is_preview: ax.set_ylabel("Count")
        elif metric == 'area':
            ax.plot(profile['bin_center'], profile['area'], marker='o', color=color, linewidth=2, markersize=3 if is_preview else 6)
            if not is_preview: ax.set_ylabel("Median Area [µm²]")
        elif metric == 'aspect':
            ax.plot(profile['bin_center'], profile['aspect'], marker='o', color=color, linewidth=2, markersize=3 if is_preview else 6)
            if not is_preview: ax.set_ylabel("Median Aspect Ratio")
            
        a_label = axis_col.split('_')[0].upper()
        
        mm_formatter = ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}")
        ax.xaxis.set_major_formatter(mm_formatter)
            
        if not is_preview:
            x_suffix = " -> top" if a_label in ['X', 'Y'] else ""
            if show_title: ax.set_title(f"{metric.capitalize()} along {a_label}-Axis")
            ax.set_xlabel(f"Position {a_label} [mm]{x_suffix}")
            
            if x_min is not None: ax.set_xlim(left=x_min * 1000.0)
            if x_max is not None: ax.set_xlim(right=x_max * 1000.0)
            PlotEngine._apply_limits(ax, None, None, y_min, y_max)
        else:
            ax.set_title(f"{metric.capitalize()} {a_label}", color="white" if dark_mode else "black", fontsize=9)
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig