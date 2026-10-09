import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from core.plot_engine import PlotEngine, lava_cmap, div_cmap, get_cmap, get_color

class ComparePlotEngine:
    @staticmethod
    def _apply_theme(fig, axes, is_preview, dark_mode, show_grid=False, label_size=10):
        bg_color = '#2b2b2b' if dark_mode else 'white'
        fg_color = 'white' if dark_mode else 'black'
        spine_color = 'gray' if dark_mode else 'black'
        
        fig.patch.set_facecolor(bg_color)
        actual_label_size = 6 if is_preview else label_size
        
        for ax in axes:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=fg_color, labelsize=actual_label_size)
            for spine in ax.spines.values(): 
                spine.set_color(spine_color)
            if not is_preview:
                ax.xaxis.label.set_color(fg_color)
                ax.yaxis.label.set_color(fg_color)
                ax.xaxis.label.set_fontsize(actual_label_size + 2)
                ax.yaxis.label.set_fontsize(actual_label_size + 2)
                ax.title.set_color(fg_color)
                ax.title.set_fontsize(actual_label_size + 2)
            if show_grid and not is_preview:
                ax.grid(True, linestyle='--', alpha=0.3, color=fg_color)

    @staticmethod
    def _setup_axes(n, is_preview, figsize, dpi, polar=False, share_x=True, share_y=True):
        fig = Figure(figsize=figsize, dpi=dpi)
        axes = []
        
        ax_kwargs = {'polar': polar}
        
        if is_preview:
            if n == 2:
                ax1 = fig.add_subplot(1, 2, 1, **ax_kwargs)
                kwargs = {**ax_kwargs}
                if share_x and not polar: kwargs['sharex'] = ax1
                if share_y and not polar: kwargs['sharey'] = ax1
                ax2 = fig.add_subplot(1, 2, 2, **kwargs)
                axes = [ax1, ax2]
            elif n == 3:
                gs = fig.add_gridspec(2, 4)
                ax1 = fig.add_subplot(gs[0, 0:2], **ax_kwargs)
                kwargs = {**ax_kwargs}
                if share_x and not polar: kwargs['sharex'] = ax1
                if share_y and not polar: kwargs['sharey'] = ax1
                ax2 = fig.add_subplot(gs[0, 2:4], **kwargs)
                ax3 = fig.add_subplot(gs[1, 1:3], **kwargs)
                axes = [ax1, ax2, ax3]
        else:
            ax1 = fig.add_subplot(n, 1, 1, **ax_kwargs)
            axes = [ax1]
            kwargs = {**ax_kwargs}
            if share_x and not polar: kwargs['sharex'] = ax1
            if share_y and not polar: kwargs['sharey'] = ax1
            for i in range(1, n):
                axes.append(fig.add_subplot(n, 1, i+1, **kwargs))
                
        return fig, axes

    # ================= DISTRIBUTIONS =================
    @staticmethod
    def create_compare_dist(loaders, col, bins=50, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        n = len(loaders)
        fig, axes = ComparePlotEngine._setup_axes(n, is_preview, figsize, dpi)
        ComparePlotEngine._apply_theme(fig, axes, is_preview, dark_mode, show_grid, label_size)
        
        global_min = min((l.df[col].min() for l in loaders if not l.df.empty), default=0)
        global_max = max((l.df[col].max() for l in loaders if not l.df.empty), default=1)
        global_bins = np.linspace(global_min, global_max, bins + 1)
        
        color = get_color(base_color)
        title = "Area" if col == 'area_um2' else "Aspect Ratio"
        
        for i, (loader, ax) in enumerate(zip(loaders, axes)):
            if not loader.df.empty:
                ax.hist(loader.df[col], bins=global_bins, color=color, edgecolor='black', alpha=0.8)
            if not is_preview:
                ax.set_title(f"{loader.sample_name}")
                ax.set_xlabel(f"{title} {'[µm²]' if col=='area_um2' else ''}")
                ax.set_ylabel("Count")
                PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
            else:
                ax.set_title(f"{i+1}", color="white" if dark_mode else "black", fontsize=8)
                
        if not is_preview and show_title: 
            fig.suptitle(f"Comparison: {title} Dist.", color='white' if dark_mode else 'black', fontsize=label_size+4, fontweight='bold')
        elif is_preview: 
            fig.suptitle(f"{title} Dist.", color="white" if dark_mode else "black", fontsize=9)
            
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    # ================= SCATTER =================
    @staticmethod
    def create_compare_scatter_corr(loaders, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        n = len(loaders)
        fig, axes = ComparePlotEngine._setup_axes(n, is_preview, figsize, dpi)
        ComparePlotEngine._apply_theme(fig, axes, is_preview, dark_mode, show_grid, label_size)
        color = get_color(base_color)
        
        for i, (loader, ax) in enumerate(zip(loaders, axes)):
            df = loader.df
            ax.scatter(df['area_um2'], df['aspect_ratio'], alpha=0.3, color=color, edgecolor='none', s=5 if is_preview else 15)
            if not is_preview:
                ax.set_title(loader.sample_name)
                ax.set_xlabel("Area [µm²]"); ax.set_ylabel("Aspect Ratio")
                PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
            else: ax.set_title(f"{i+1}", color="white" if dark_mode else "black", fontsize=8)
            
        if not is_preview and show_title: fig.suptitle("Area vs. Aspect Ratio", color='white' if dark_mode else 'black', fontsize=label_size+4, fontweight='bold')
        elif is_preview: fig.suptitle("Scatter", color="white" if dark_mode else "black", fontsize=9)
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    @staticmethod
    def create_compare_scatter_tiles(loaders, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        n = len(loaders)
        fig, axes = ComparePlotEngine._setup_axes(n, is_preview, figsize, dpi)
        ComparePlotEngine._apply_theme(fig, axes, is_preview, dark_mode, show_grid, label_size)
        color = get_color(base_color)
        
        for i, (loader, ax) in enumerate(zip(loaders, axes)):
            df = loader.df
            if not df.empty:
                _, _, dens, area, _ = PlotEngine._generate_2d_grids(df, loader.sample_height_um, 1, 3)
                d_flat, a_flat = dens.flatten(), area.flatten()
                m = ~np.isnan(d_flat) & ~np.isnan(a_flat) & (d_flat > 0)
                ax.scatter(d_flat[m], a_flat[m], alpha=0.7, color=color, edgecolor='black', linewidth=0.5, s=5 if is_preview else 20)
                
            if not is_preview:
                fg = 'white' if dark_mode else 'black'
                ax.axvline(loader.global_density, color=fg, linestyle='--', alpha=0.5)
                ax.axhline(loader.global_med_area, color=fg, linestyle=':', alpha=0.5)
                ax.set_title(loader.sample_name)
                ax.set_xlabel("Local Density [hyd/mm³]"); ax.set_ylabel("Local Median Area [µm²]")
                PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
            else: ax.set_title(f"{i+1}", color="white" if dark_mode else "black", fontsize=8)
            
        if not is_preview and show_title: fig.suptitle("Tile Scatter", color='white' if dark_mode else 'black', fontsize=label_size+4, fontweight='bold')
        elif is_preview: fig.suptitle("Tile Scatter", color="white" if dark_mode else "black", fontsize=9)
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    # ================= HEATMAPS =================
    @staticmethod
    def create_compare_heatmap(loaders, map_type, tile_mult=1, blur_val=3, dark_mode=True, show_title=True, show_grid=False, label_size=10, cmap_name='lava', base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(6, 5), dpi=100, is_preview=False, **kwargs):
        n = len(loaders)
        fig, axes = ComparePlotEngine._setup_axes(n, is_preview, figsize, dpi, share_y=False)
        ComparePlotEngine._apply_theme(fig, axes, is_preview, dark_mode, show_grid, label_size)
        
        grid_data = []
        for loader in loaders:
            if loader.df.empty: 
                grid_data.append((None, None, None))
                continue
            x_edges, y_edges, dens, area, aspect = PlotEngine._generate_2d_grids(loader.df, loader.sample_height_um, tile_mult, blur_val)
            if map_type == "Density Heatmap": grid, label = dens, "Density"
            elif map_type == "Area Heatmap": grid, label = area, "Area"
            elif map_type == "Aspect Heatmap": grid, label = aspect, "Aspect"
            elif map_type == "Relation Heatmap":
                # Suppress divide-by-zero warnings when density is zero.
                with np.errstate(divide='ignore', invalid='ignore'):
                    R = (area / loader.global_med_area) * (loader.global_density / dens)
                    grid = np.full_like(R, np.nan)
                    m_pos = (R > 1) & ~np.isnan(R) & ~np.isinf(R)
                    grid[m_pos] = R[m_pos] - 1.0
                    m_neg = (R <= 1) & (R > 0) & ~np.isnan(R) & ~np.isinf(R)
                    grid[m_neg] = -(1.0 / R[m_neg]) + 1.0
                label = "Relation"
            else: 
                d_flat, a_flat = dens.flatten(), area.flatten()
                m = ~np.isnan(d_flat) & ~np.isnan(a_flat) & (d_flat > 0)
                d_val, a_val = d_flat[m], a_flat[m]
                if len(d_val) == 0: 
                    grid_data.append((None, None, None)); continue
                window_vol = (1500.0 / 1000.0)**2 * (loader.sample_height_um / 1000.0)
                step_d = 1.0 / window_vol
                max_c = np.ceil(np.max(d_val) / step_d)
                x_e = np.arange(0.5 * step_d, (max_c + 1.5) * step_d, step_d)
                y_e = np.arange(0, np.max(a_val) + 4000, 2000.0)
                H, xe, ye = np.histogram2d(d_val, a_val, bins=[x_e, y_e])
                grid = np.ma.masked_where(H == 0, H).T
                x_edges, y_edges = xe, ye
                label = "Tiles"
                
            grid_data.append((x_edges, y_edges, grid))
            
        all_grids = [g[2] for g in grid_data if g[2] is not None]
        if not all_grids: return fig
        
        cmap = get_cmap(cmap_name)
        if map_type == "Relation Heatmap":
            max_abs = max(np.max(np.abs(g[~np.isnan(g) & ~np.isinf(g)])) for g in all_grids if np.any(~np.isnan(g)))
            if max_abs == 0: max_abs = 1.0
            vmin, vmax = -max_abs, max_abs
            if cmap_name == 'lava': cmap = div_cmap
        else:
            vmin = min(np.nanmin(g) for g in all_grids if np.any(~np.isnan(g)))
            vmax = max(np.nanmax(g) for g in all_grids if np.any(~np.isnan(g)))

        mm_formatter = ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}")
        for i, (ax, (xe, ye, grid), loader) in enumerate(zip(axes, grid_data, loaders)):
            if grid is not None:
                mesh = ax.pcolormesh(xe, ye, grid, cmap=cmap, vmin=vmin, vmax=vmax)
                if "Tile" not in map_type:
                    ax.set_aspect('equal', adjustable='box')
                    
                    x_dmin, x_dmax = xe.min(), xe.max()
                    y_dmin, y_dmax = ye.min(), ye.max()
                    
                    ax.xaxis.set_major_formatter(mm_formatter)
                    ax.yaxis.set_major_formatter(mm_formatter)
                    
                    if not is_preview:
                        ax.set_xlabel("X [mm] -> top")
                        ax.set_ylabel("Y [mm]")
                        if x_min is not None: x_dmin = x_min * 1000.0
                        if x_max is not None: x_dmax = x_max * 1000.0
                        if y_min is not None: y_dmin = y_min * 1000.0
                        if y_max is not None: y_dmax = y_max * 1000.0
                        
                    # Reverse the Y axis and apply the selected limits.
                    ax.set_xlim(left=x_dmin, right=x_dmax)
                    ax.set_ylim(bottom=y_dmax, top=y_dmin)
                else:
                    if not is_preview:
                        ax.set_xlabel("Local Density [hyd/mm³]")
                        ax.set_ylabel("Local Median Area [µm²]")
                        PlotEngine._apply_limits(ax, x_min, x_max, y_min, y_max)
                        
            if not is_preview: ax.set_title(loader.sample_name)
            else: ax.set_title(f"{i+1}", color="white" if dark_mode else "black", fontsize=8)

        if not is_preview:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            cb = fig.colorbar(mesh, cax=cbar_ax)
            fg = 'white' if dark_mode else 'black'
            cb.set_label(label, color=fg, fontsize=label_size)
            cb.ax.yaxis.set_tick_params(color=fg, labelcolor=fg, labelsize=label_size)
            if show_title: fig.suptitle(f"Comparison: {map_type}", color=fg, fontsize=label_size+4, fontweight='bold')
        else:
            fig.suptitle(map_type, color="white" if dark_mode else "black", fontsize=9)
            
        fig.tight_layout(pad=0.5 if is_preview else 1.0, rect=[0, 0, 0.85, 1] if not is_preview else [0, 0, 1, 1])
        return fig

    # ================= PROFILES =================
    @staticmethod
    def create_compare_profile(loaders, axis_col, metric, bin_width=500.0, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        n = len(loaders)
        share_x_flag = False if axis_col == 'y_um' else True
        fig, axes = ComparePlotEngine._setup_axes(n, is_preview, figsize, dpi, share_x=share_x_flag)
        ComparePlotEngine._apply_theme(fig, axes, is_preview, dark_mode, show_grid, label_size)
        
        global_min = min((l.df[axis_col].min() for l in loaders if not l.df.empty), default=0)
        global_max = max((l.df[axis_col].max() for l in loaders if not l.df.empty), default=1)
        
        min_v, max_v = np.floor(global_min/bin_width)*bin_width, np.ceil(global_max/bin_width)*bin_width
        bins = np.arange(min_v, max_v + bin_width, bin_width)
        
        mm_formatter = ticker.FuncFormatter(lambda x, pos: f"{x/1000:g}")
        
        color = get_color(base_color)
        if base_color == 'Blue' and metric == 'count': color = '#34495e'
        elif base_color == 'Blue' and metric == 'area': color = '#e74c3c'
        elif base_color == 'Blue' and metric == 'aspect': color = '#f39c12'
        
        a_label = axis_col.split('_')[0].upper()
        x_suffix = " -> top" if a_label in ['X', 'Y'] else ""
        
        for i, (loader, ax) in enumerate(zip(loaders, axes)):
            if loader.df.empty: continue
            
            # Normalize coordinates consistently with PlotEngine.
            offset = loader.df[axis_col].min()
            df_copy = loader.df.copy()
            df_copy[axis_col] = df_copy[axis_col] - offset
            
            df_copy['bin'] = pd.cut(df_copy[axis_col], bins=bins)
            prof = df_copy.groupby('bin', observed=False).agg(count=(axis_col, 'size'), area=('area_um2', 'median'), aspect=('aspect_ratio', 'median')).reset_index()
            prof['bin_center'] = prof['bin'].apply(lambda x: x.mid).astype(float)
            
            if metric == 'count': 
                ax.bar(prof['bin_center'], prof['count'], width=bin_width*0.9, color=color, edgecolor='black')
                if not is_preview: ax.set_ylabel("Count")
            elif metric == 'area': 
                ax.plot(prof['bin_center'], prof['area'], marker='o', color=color)
                if not is_preview: ax.set_ylabel("Median Area [µm²]")
            elif metric == 'aspect': 
                ax.plot(prof['bin_center'], prof['aspect'], marker='o', color=color)
                if not is_preview: ax.set_ylabel("Median Aspect Ratio")
            
            ax.xaxis.set_major_formatter(mm_formatter)
            
            if not is_preview:
                ax.set_title(loader.sample_name)
                ax.set_xlabel(f"Position {a_label} [mm]{x_suffix}")
                if x_min is not None: ax.set_xlim(left=x_min * 1000.0)
                if x_max is not None: ax.set_xlim(right=x_max * 1000.0)
                PlotEngine._apply_limits(ax, None, None, y_min, y_max)
            else: ax.set_title(f"{i+1}", color="white" if dark_mode else "black", fontsize=8)
            
        if not is_preview and show_title: fig.suptitle(f"Compare: {metric.capitalize()} along {a_label}", color='white' if dark_mode else 'black', fontsize=label_size+4, fontweight='bold')
        elif is_preview: fig.suptitle(f"{metric.capitalize()} {a_label}", color="white" if dark_mode else "black", fontsize=9)
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    # ================= ROSE DIAGRAMS =================
    @staticmethod
    def create_compare_rose(loaders, is_half=False, dark_mode=True, show_title=True, show_grid=False, label_size=10, base_color='Red', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        n = len(loaders)
        fig, axes = ComparePlotEngine._setup_axes(n, is_preview, figsize, dpi, polar=True)
        ComparePlotEngine._apply_theme(fig, axes, is_preview, dark_mode, show_grid, label_size)
        
        global_max_r = 1.0 if is_half else 0.0
        for loader in loaders:
            if not loader.df.empty:
                if is_half:
                    m = loader.df['aspect_ratio'].max()
                    if m > global_max_r: global_max_r = m
                else:
                    angles = np.deg2rad(loader.df['angle_deg'])
                    angles = np.concatenate([angles, angles + np.pi])
                    counts, _ = np.histogram(angles, bins=np.linspace(0, 2*np.pi, 37))
                    m = np.max(counts)
                    if m > global_max_r: global_max_r = m

        color = get_color(base_color)
        
        # Adjust tick-label padding to accommodate the selected label size.
        pad_val = 5 if is_preview else max(10, label_size + 5)
        
        for i, (loader, ax) in enumerate(zip(loaders, axes)):
            df = loader.df
            if not df.empty:
                if not is_half:
                    angles = np.deg2rad(df['angle_deg'])
                    angles = np.concatenate([angles, angles + np.pi])
                    counts, bins = np.histogram(angles, bins=np.linspace(0, 2*np.pi, 37))
                    ax.bar(bins[:-1], counts, width=2*np.pi/36, bottom=0.0, color=color, edgecolor='black', alpha=0.7)
                else:
                    c = color if base_color != 'Red' else '#2980b9'
                    ax.scatter(np.deg2rad(df['angle_deg'] % 180), df['aspect_ratio'], alpha=0.6, color=c, s=5 if is_preview else 15)

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            
            if is_half:
                ax.set_rmax(global_max_r)
                ax.set_rorigin(1.0)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.set_thetamin(0)
                ax.set_thetamax(180)
                ax.set_xticks(np.deg2rad([0, 90, 180]))
                ax.set_xticklabels(['-y\n[0°]', '+x\n[90°]', '+y\n[180°]'])
            else:
                ax.set_rmax(global_max_r)
                ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
                ax.set_xticklabels(['-y\n[0°]', '+x\n[90°]', '+y\n[180°]', '-x\n[270°]'])

            # Apply the selected tick-label padding.
            ax.tick_params(axis='x', pad=pad_val)

            if not is_preview: 
                ax.set_title(loader.sample_name, pad=25 + pad_val)
                if y_min is not None: ax.set_rorigin(y_min)
                if y_max is not None: ax.set_rmax(y_max)
            else:
                ax.set_title(f"{i+1}", color="white" if dark_mode else "black", fontsize=8, pad=5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
        title = "Half-Rose" if is_half else "Rose Diagram"
        if not is_preview and show_title: fig.suptitle(f"Comparison: {title}", color='white' if dark_mode else 'black', fontsize=label_size+4, fontweight='bold')
        elif is_preview: fig.suptitle(title, color="white" if dark_mode else "black", fontsize=9)
        
        fig.tight_layout(pad=0.5 if is_preview else 1.0)
        return fig

    @staticmethod
    def create_compare_half_rose_heatmap(loaders, angle_bin_deg=5, aspect_bin_num=5, dark_mode=True, show_title=True, show_grid=False, label_size=10, cmap_name='lava', base_color='Blue', x_min=None, x_max=None, y_min=None, y_max=None, figsize=(5, 4), dpi=100, is_preview=False, **kwargs):
        n = len(loaders)
        fig, axes = ComparePlotEngine._setup_axes(n, is_preview, figsize, dpi, polar=True)
        ComparePlotEngine._apply_theme(fig, axes, is_preview, dark_mode, show_grid, label_size)
        
        grid_data = []
        global_max_r = 1.0
        for loader in loaders:
            df = loader.df
            if df.empty:
                grid_data.append((None, None, None))
                continue
            angles = df['angle_deg'] % 180
            aspects = df['aspect_ratio']
            aspect_bin = aspect_bin_num / 30.0

            theta_edges = np.deg2rad(np.arange(0, 180 + angle_bin_deg, angle_bin_deg))
            r_max = np.max(aspects)
            if r_max > global_max_r: global_max_r = r_max
            r_edges = np.arange(1.0, r_max + aspect_bin * 2, aspect_bin)

            H, t_edges, r_edges = np.histogram2d(np.deg2rad(angles), aspects, bins=[theta_edges, r_edges])
            H_masked = np.ma.masked_where(H == 0, H)
            T, R = np.meshgrid(t_edges, r_edges)
            grid_data.append((T, R, H_masked.T))

        all_grids = [g[2] for g in grid_data if g[2] is not None]
        vmax = max((np.max(g) for g in all_grids), default=1)

        cmap = get_cmap(cmap_name)
        mesh = None
        
        # Adjust tick-label padding to accommodate the selected label size.
        pad_val = 5 if is_preview else max(10, label_size + 5)
        
        for i, (ax, (T, R, grid), loader) in enumerate(zip(axes, grid_data, loaders)):
            if grid is not None:
                mesh = ax.pcolormesh(T, R, grid, cmap=cmap, vmin=0, vmax=vmax)

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rorigin(1.0)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_rmax(global_max_r)
            
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_xticks(np.deg2rad([0, 90, 180]))
            ax.set_xticklabels(['-y\n[0°]', '+x\n[90°]', '+y\n[180°]'])

            # Apply the selected tick-label padding.
            ax.tick_params(axis='x', pad=pad_val)

            if not is_preview: 
                ax.set_title(loader.sample_name, pad=25 + pad_val)
                if y_min is not None: ax.set_rorigin(y_min)
                if y_max is not None: ax.set_rmax(y_max)
            else:
                ax.set_title(f"{i+1}", color="white" if dark_mode else "black", fontsize=8, pad=5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        if not is_preview and mesh is not None:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            cb = fig.colorbar(mesh, cax=cbar_ax)
            fg = 'white' if dark_mode else 'black'
            cb.set_label("Count", color=fg, fontsize=label_size)
            cb.ax.yaxis.set_tick_params(color=fg, labelcolor=fg, labelsize=label_size)
            if show_title: fig.suptitle("Compare: Half-Rose Heatmap", color=fg, fontsize=label_size+4, fontweight='bold')
        else:
            fig.suptitle("Half-Rose Heat", color="white" if dark_mode else "black", fontsize=9)

        fig.tight_layout(pad=0.5 if is_preview else 1.0, rect=[0, 0, 0.85, 1] if not is_preview else [0, 0, 1, 1])
        return fig