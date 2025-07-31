#!/usr/bin/env python3
"""
ASL Landmark Visualizer

A comprehensive visualization tool for American Sign Language (ASL) landmarks from the
Google Isolated Sign Language Recognition dataset. This script provides multiple ways
to visualize sign language gestures including animations, static mean frames, and
interactive dashboards.

Features:
    - Animated 2D visualizations with landmark connections
    - Static mean frame visualizations showing average positions
    - GIF export for animations (matplotlib-based)
    - HTML export for interactive Plotly visualizations
    - Interactive dashboard mode with sign/sequence selection
    - Color-coded landmarks: red (face), blue (hands), green (pose)
    - Automatic handling of missing landmarks

Basic Usage Examples:
    # View an animation in browser
    python visualizer.py --sequence_id 1002052130 --show
    
    # Save animation as GIF
    python visualizer.py --sequence_id 1002052130 --output tv_sign.gif
    
    # Save animation as interactive HTML
    python visualizer.py --sequence_id 1002052130 --output tv_sign.html
    
    # Visualize by sign name (uses first sequence)
    python visualizer.py --sign "blow" --output blow.gif

Advanced Usage Examples:
    # High-quality GIF with custom settings
    python visualizer.py --sequence_id 1002052130 --output hq_sign.gif --fps 20 --width 10 --height 8
    
    # Create mean frame visualization
    python visualizer.py --sign "cloud" --mode mean --output cloud_mean.html
    
    # Generate both animation and mean frame
    python visualizer.py --sequence_id 1187993400 --mode both --output outputs/analysis
    
    # Launch interactive dashboard
    python visualizer.py --dashboard
    
    # Use custom data directory
    python visualizer.py --data_dir /path/to/asl/data --sequence_id 1002052130 --show
    
    # Dashboard on custom port
    python visualizer.py --dashboard --port 8080

Output Formats:
    - .gif: Animated GIF using matplotlib (requires matplotlib, PIL)
    - .html: Interactive Plotly visualization
    - Dashboard: Real-time web interface (requires dash, jupyter_dash)

Dependencies:
    Required: pandas, plotly, pyarrow
    For GIF: matplotlib, Pillow
    For Dashboard: dash, jupyter_dash, dash-bootstrap-components
"""

import argparse
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from typing import Optional, Tuple, List
import warnings
import numpy as np
try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore")

pio.templates.default = "simple_white"


class ASLVisualizer:
    """Visualizer for ASL landmark data"""
    
    def __init__(self, data_directory: str = 'data/'):
        """
        Initialize the visualizer with data directory.
        
        Args:
            data_directory: Path to the directory containing train.csv and parquet files
        """
        self.data_directory = data_directory
        self.train_data = None
        self.connections = self._define_connections()
        self._load_train_data()
    
    def _load_train_data(self):
        """Load the train.csv file"""
        train_csv_path = os.path.join(self.data_directory, 'train.csv')
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"train.csv not found in {self.data_directory}")
        self.train_data = pd.read_csv(train_csv_path)
    
    @staticmethod
    def _define_connections() -> List[List[int]]:
        """Define the connections between landmarks"""
        return [
            # right hand
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20],
            # pose
            [38, 36, 35, 34, 30, 31, 32, 33, 37],
            [40, 39],
            [52, 46, 50, 48, 46, 44, 42, 41, 43, 45, 47, 49, 45, 51],
            [42, 54, 56, 58, 60, 62, 58],
            [41, 53, 55, 57, 59, 61, 57],
            [54, 53],
            # left hand
            [80, 81, 82, 83, 84],
            [80, 85, 86, 87, 88],
            [80, 89, 90, 91, 92],
            [80, 93, 94, 95, 96],
            [80, 97, 98, 99, 100],
        ]
    
    @staticmethod
    def _assign_color(landmark_type: str) -> str:
        """Assign color based on landmark type"""
        if landmark_type == 'face':
            return 'red'
        elif 'hand' in landmark_type:
            return 'dodgerblue'
        else:
            return 'green'
    
    @staticmethod
    def _assign_order(row) -> int:
        """Assign plotting order based on landmark type"""
        if row.type == 'face':
            return row.landmark_index + 101
        elif row.type == 'pose':
            return row.landmark_index + 30
        elif row.type == 'left_hand':
            return row.landmark_index + 80
        else:
            return row.landmark_index
    
    def get_parquet_path(self, sequence_id: int) -> Tuple[str, str]:
        """
        Get parquet file path and sign category for a sequence ID.
        
        Args:
            sequence_id: The sequence ID to look up
            
        Returns:
            Tuple of (parquet_path, sign_category)
        """
        parquet_file = self.train_data[self.train_data.sequence_id == sequence_id]
        if parquet_file.empty:
            raise ValueError(f"Sequence ID {sequence_id} not found")
        
        # Path in train.csv already includes 'data/' prefix, so use it directly
        parquet_path = parquet_file.path.values[0]
        sign_cat = parquet_file.sign.values[0]
        return parquet_path, sign_cat
    
    def get_sequences_by_sign(self, sign: str) -> List[int]:
        """Get all sequence IDs for a given sign"""
        sequences = self.train_data[self.train_data.sign == sign]['sequence_id'].tolist()
        if not sequences:
            raise ValueError(f"Sign '{sign}' not found")
        return sequences
    
    def visualize_2d_animation(self, parquet_df: pd.DataFrame, title: str = "ASL Sign Visualization") -> go.Figure:
        """
        Create an animated 2D visualization of landmarks.
        
        Args:
            parquet_df: DataFrame containing landmark data
            title: Title for the visualization
            
        Returns:
            Plotly figure object
        """
        frames = sorted(set(parquet_df.frame))
        first_frame = min(frames)
        
        parquet_df['color'] = parquet_df.type.apply(lambda row: self._assign_color(row))
        parquet_df['plot_order'] = parquet_df.apply(lambda row: self._assign_order(row), axis=1)
        
        first_frame_df = parquet_df[parquet_df.frame == first_frame].copy()
        first_frame_df = first_frame_df.sort_values(["plot_order"]).set_index('plot_order')
        
        frames_list = []
        for frame in frames:
            filtered_df = parquet_df[parquet_df.frame == frame].copy()
            filtered_df = filtered_df.sort_values(["plot_order"]).set_index("plot_order")
            
            traces = [go.Scatter(
                x=filtered_df['x'],
                y=filtered_df['y'],
                mode='markers',
                marker=dict(
                    color=filtered_df.color,
                    size=9
                )
            )]
            
            for seg in self.connections:
                trace = go.Scatter(
                    x=filtered_df.loc[seg]['x'],
                    y=filtered_df.loc[seg]['y'],
                    mode='lines',
                    line=dict(color='black', width=2)
                )
                traces.append(trace)
            
            frame_data = go.Frame(data=traces, traces=[i for i in range(len(traces))])
            frames_list.append(frame_data)
        
        # Initial frame
        traces = [go.Scatter(
            x=first_frame_df['x'],
            y=first_frame_df['y'],
            mode='markers',
            marker=dict(
                color=first_frame_df.color,
                size=9
            )
        )]
        
        for seg in self.connections:
            trace = go.Scatter(
                x=first_frame_df.loc[seg]['x'],
                y=first_frame_df.loc[seg]['y'],
                mode='lines',
                line=dict(color='black', width=2)
            )
            traces.append(trace)
        
        fig = go.Figure(data=traces, frames=frames_list)
        
        # Layout
        fig.update_layout(
            title=title,
            width=1000,
            height=800,
            updatemenus=[{
                "buttons": [{
                    "args": [None, {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }],
                    "label": "&#9654;",
                    "method": "animate",
                }],
                "direction": "left",
                "pad": {"r": 100, "t": 100},
                "font": {"size": 30},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }],
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, autorange="reversed")
        )
        
        return fig
    
    def visualize_mean_frame(self, parquet_df: pd.DataFrame, title: str = "ASL Sign Mean Frame") -> go.Figure:
        """
        Create a visualization of mean landmark positions.
        
        Args:
            parquet_df: DataFrame containing landmark data
            title: Title for the visualization
            
        Returns:
            Plotly figure object
        """
        parquet_df['color'] = parquet_df.type.apply(lambda row: self._assign_color(row))
        parquet_df['plot_order'] = parquet_df.apply(lambda row: self._assign_order(row), axis=1)
        
        # Calculate mean positions
        mean_df = parquet_df.groupby("plot_order").agg({
            'x': 'mean',
            'y': 'mean',
            'color': 'first'
        }).reset_index()
        mean_df = mean_df.sort_values(["plot_order"]).set_index('plot_order')
        
        traces = [go.Scatter(
            x=mean_df['x'],
            y=mean_df['y'],
            mode='markers',
            marker=dict(
                color=mean_df.color,
                size=9
            )
        )]
        
        for seg in self.connections:
            trace = go.Scatter(
                x=mean_df.loc[seg]['x'],
                y=mean_df.loc[seg]['y'],
                mode='lines',
                line=dict(color='black', width=2)
            )
            traces.append(trace)
        
        fig = go.Figure(data=traces)
        
        # Layout
        fig.update_layout(
            title=title,
            width=1000,
            height=800,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, autorange="reversed")
        )
        
        return fig
    
    def save_as_gif(self, parquet_df: pd.DataFrame, output_path: str, 
                    title: str = "ASL Sign Visualization", fps: int = 10,
                    width: int = 8, height: int = 6) -> None:
        """
        Save animation as GIF file using matplotlib.
        
        Args:
            parquet_df: DataFrame containing landmark data
            output_path: Path to save the GIF file
            title: Title for the visualization
            fps: Frames per second for the GIF
            width: Width of the figure in inches
            height: Height of the figure in inches
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for GIF export. Install with: pip install matplotlib")
        
        frames = sorted(set(parquet_df.frame))
        parquet_df['color'] = parquet_df.type.apply(lambda row: self._assign_color(row))
        parquet_df['plot_order'] = parquet_df.apply(lambda row: self._assign_order(row), axis=1)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set fixed limits based on data range
        all_x = parquet_df['x'].dropna()
        all_y = parquet_df['y'].dropna()
        x_margin = (all_x.max() - all_x.min()) * 0.1
        y_margin = (all_y.max() - all_y.min()) * 0.1
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.max() + y_margin, all_y.min() - y_margin)  # Reversed for y
        
        # Initialize line objects
        scatter_plots = []
        line_plots = []
        
        def animate(frame_idx):
            # Clear previous plots
            for sp in scatter_plots:
                sp.remove()
            for lp in line_plots:
                lp.remove()
            scatter_plots.clear()
            line_plots.clear()
            
            # Get data for current frame
            frame_num = frames[frame_idx]
            filtered_df = parquet_df[parquet_df.frame == frame_num].copy()
            filtered_df = filtered_df.sort_values(["plot_order"]).set_index("plot_order")
            
            # Plot landmarks by type
            for landmark_type in ['face', 'pose', 'left_hand', 'right_hand']:
                type_df = filtered_df[filtered_df.index.isin(
                    [i for i in filtered_df.index if self._get_landmark_type(i) == landmark_type]
                )]
                if not type_df.empty:
                    color = self._assign_color(landmark_type)
                    sp = ax.scatter(type_df['x'], type_df['y'], 
                                  c=color, s=50, alpha=0.8)
                    scatter_plots.append(sp)
            
            # Plot connections
            for seg in self.connections:
                try:
                    x_vals = filtered_df.loc[seg]['x'].values
                    y_vals = filtered_df.loc[seg]['y'].values
                    if len(x_vals) == len(seg):  # Ensure all points exist
                        lp, = ax.plot(x_vals, y_vals, 'k-', linewidth=2, alpha=0.6)
                        line_plots.append(lp)
                except (KeyError, AttributeError):
                    continue
            
            return scatter_plots + line_plots
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames),
            interval=1000/fps, blit=True
        )
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        
        print(f"Saved animation to {output_path}")
    
    def _get_landmark_type(self, idx: int) -> str:
        """Determine landmark type from index."""
        if idx <= 20:
            return 'right_hand'
        elif 30 <= idx <= 62:
            return 'pose'
        elif 80 <= idx <= 100:
            return 'left_hand'
        else:
            return 'face'
    
    def create_dashboard(self, port: int = 8050):
        """Create interactive dashboard (requires dash and jupyter_dash)"""
        try:
            from jupyter_dash import JupyterDash
            from dash import dcc, html
            from dash.dependencies import Input, Output
            import dash_bootstrap_components as dbc
        except ImportError:
            print("Dashboard mode requires: pip install jupyter_dash dash dash-bootstrap-components")
            return
        
        app = JupyterDash(__name__, 
                         meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                         external_stylesheets=[dbc.themes.BOOTSTRAP],
                         suppress_callback_exceptions=True)
        
        # Create dropdown options
        dropdown_options = self.train_data.groupby('sign')['sequence_id'].apply(list).to_dict()
        names = list(dropdown_options.keys())
        
        sign_dropdown = dcc.Dropdown(
            options=[{'label': name, 'value': name} for name in names],
            id='sign_dropdown',
            clearable=False,
            value=names[0],
            className="dbc",
            placeholder='Select a Sign',
            maxHeight=200
        )
        
        sequence_dropdown = dcc.Dropdown(
            id='sequence_dropdown',
            clearable=False,
            className="dbc",
            placeholder='Select a Sequence ID',
            maxHeight=200
        )
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("ASL Landmark Visualizer"), width=12)
            ]),
            dbc.Row([
                dbc.Col(sign_dropdown, width=6),
                dbc.Col(sequence_dropdown, width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.RadioItems(
                        id='mode_selector',
                        options=[
                            {'label': 'Animation', 'value': 'animation'},
                            {'label': 'Mean Frame', 'value': 'mean'}
                        ],
                        value='animation',
                        inline=True
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2(id='sign_cat'),
                    dcc.Graph(id='fig', style={'height': 800}),
                ], width=12)
            ])
        ])
        
        @app.callback(
            Output('sequence_dropdown', 'options'),
            [Input('sign_dropdown', 'value')]
        )
        def update_dropdown(name):
            return [{'label': i, 'value': i} for i in dropdown_options[name]]
        
        @app.callback(
            [Output('fig', 'figure'),
             Output('sign_cat', 'children')],
            [Input('sequence_dropdown', 'value'),
             Input('mode_selector', 'value')]
        )
        def update_visualization(sequence_id, mode):
            if not sequence_id:
                return {}, ""
            
            parquet_path, sign_cat = self.get_parquet_path(sequence_id)
            parquet_df = pd.read_parquet(parquet_path)
            
            if mode == 'animation':
                fig = self.visualize_2d_animation(parquet_df, f"{sign_cat} - Sequence {sequence_id}")
            else:
                fig = self.visualize_mean_frame(parquet_df, f"{sign_cat} - Mean Frame")
            
            return fig, f"Sign: {sign_cat}"
        
        print(f"Starting dashboard on http://localhost:{port}")
        app.run_server(port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description='Visualize ASL landmark data')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory containing train.csv and parquet files')
    parser.add_argument('--sequence_id', type=int, help='Sequence ID to visualize')
    parser.add_argument('--sign', type=str, help='Sign category to visualize (will use first sequence)')
    parser.add_argument('--mode', choices=['animation', 'mean', 'both'], default='animation',
                        help='Visualization mode')
    parser.add_argument('--output', type=str, help='Output file path (HTML or GIF based on extension)')
    parser.add_argument('--show', action='store_true', help='Show visualization in browser')
    parser.add_argument('--dashboard', action='store_true', help='Launch interactive dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port for dashboard (default: 8050)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for GIF export (default: 10)')
    parser.add_argument('--width', type=int, default=8, help='Width for GIF export in inches (default: 8)')
    parser.add_argument('--height', type=int, default=6, help='Height for GIF export in inches (default: 6)')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ASLVisualizer(args.data_dir)
    
    # Dashboard mode
    if args.dashboard:
        visualizer.create_dashboard(args.port)
        return
    
    # Get sequence ID
    if args.sequence_id:
        sequence_id = args.sequence_id
    elif args.sign:
        sequences = visualizer.get_sequences_by_sign(args.sign)
        sequence_id = sequences[0]
        print(f"Using sequence {sequence_id} for sign '{args.sign}'")
    else:
        print("Please specify either --sequence_id or --sign")
        return
    
    # Load parquet data
    parquet_path, sign_cat = visualizer.get_parquet_path(sequence_id)
    parquet_df = pd.read_parquet(parquet_path)
    
    # Create visualizations
    if args.mode == 'animation':
        # Check if output is GIF
        if args.output and args.output.lower().endswith('.gif'):
            visualizer.save_as_gif(parquet_df, args.output, 
                                 f"Sign: {sign_cat}",
                                 fps=args.fps, width=args.width, height=args.height)
        else:
            fig = visualizer.visualize_2d_animation(parquet_df, f"Sign: {sign_cat}")
            if args.output:
                fig.write_html(args.output)
                print(f"Saved visualization to {args.output}")
            if args.show or not args.output:
                fig.show()
                
    elif args.mode == 'mean':
        fig = visualizer.visualize_mean_frame(parquet_df, f"Sign: {sign_cat} (Mean Frame)")
        if args.output:
            if args.output.lower().endswith('.gif'):
                print("GIF export is only available for animation mode, not mean frame")
                return
            fig.write_html(args.output)
            print(f"Saved visualization to {args.output}")
        if args.show or not args.output:
            fig.show()
            
    else:  # both
        if args.output:
            base, ext = os.path.splitext(args.output)
            
            # Handle GIF export for animation
            if ext.lower() == '.gif':
                visualizer.save_as_gif(parquet_df, f"{base}_animation.gif",
                                     f"Sign: {sign_cat}",
                                     fps=args.fps, width=args.width, height=args.height)
                # Save mean frame as HTML
                fig2 = visualizer.visualize_mean_frame(parquet_df, f"Sign: {sign_cat} (Mean Frame)")
                fig2.write_html(f"{base}_mean.html")
                print(f"Saved to {base}_animation.gif and {base}_mean.html")
            else:
                # Save both as HTML
                fig1 = visualizer.visualize_2d_animation(parquet_df, f"Sign: {sign_cat}")
                fig2 = visualizer.visualize_mean_frame(parquet_df, f"Sign: {sign_cat} (Mean Frame)")
                fig1.write_html(f"{base}_animation{ext}")
                fig2.write_html(f"{base}_mean{ext}")
                print(f"Saved to {base}_animation{ext} and {base}_mean{ext}")
        
        if args.show:
            fig1 = visualizer.visualize_2d_animation(parquet_df, f"Sign: {sign_cat}")
            fig2 = visualizer.visualize_mean_frame(parquet_df, f"Sign: {sign_cat} (Mean Frame)")
            fig1.show()
            fig2.show()


if __name__ == "__main__":
    main()