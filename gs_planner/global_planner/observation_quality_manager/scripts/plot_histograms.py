#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
from threading import Lock

class HistogramPlotter:
    def __init__(self):
        rospy.init_node('histogram_plotter', anonymous=True)

        # Data storage with thread locks
        self.geometry_data = []
        self.texture_data = []
        self.observation_score_data = []

        self.geometry_lock = Lock()
        self.texture_lock = Lock()
        self.observation_score_lock = Lock()

        # Subscribers
        rospy.Subscriber('/geometry_complexity_data', Float32MultiArray,
                         self.geometry_callback, queue_size=1)
        rospy.Subscriber('/texture_complexity_data', Float32MultiArray,
                         self.texture_callback, queue_size=1)
        rospy.Subscriber('/observation_score_data', Float32MultiArray,
                         self.observation_score_callback, queue_size=1)

        # Setup matplotlib
        plt.ion()  # Enable interactive mode
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle('Observation Quality Manager - Histogram Analysis', fontsize=14, fontweight='bold')

        # Configure subplots
        self.axes[0].set_title('Geometric Complexity')
        self.axes[0].set_xlabel('Complexity Value')
        self.axes[0].set_ylabel('Count')
        self.axes[0].grid(True, alpha=0.3)

        self.axes[1].set_title('Texture Complexity')
        self.axes[1].set_xlabel('Complexity Value')
        self.axes[1].set_ylabel('Count')
        self.axes[1].grid(True, alpha=0.3)

        self.axes[2].set_title('Observation Score')
        self.axes[2].set_xlabel('Score Value')
        self.axes[2].set_ylabel('Count')
        self.axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        rospy.loginfo("Histogram Plotter initialized. Waiting for data...")

    def geometry_callback(self, msg):
        with self.geometry_lock:
            self.geometry_data = list(msg.data)

    def texture_callback(self, msg):
        with self.texture_lock:
            self.texture_data = list(msg.data)

    def observation_score_callback(self, msg):
        with self.observation_score_lock:
            self.observation_score_data = list(msg.data)

    def plot_histogram(self, ax, data, bins, range_tuple, color, title_suffix=""):
        """Plot histogram with statistics"""
        ax.clear()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No Data',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=12)
            ax.set_xlim(range_tuple)
            return

        # Calculate statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)
        min_val = np.min(data)
        max_val = np.max(data)

        # Plot histogram
        n, bins_edges, patches = ax.hist(data, bins=bins, range=range_tuple,
                                          color=color, alpha=0.7, edgecolor='black')

        # Add statistics text
        stats_text = f'N: {len(data)}\n'
        stats_text += f'Mean: {mean_val:.4f}\n'
        stats_text += f'Std: {std_val:.4f}\n'
        stats_text += f'Median: {median_val:.4f}\n'
        stats_text += f'Min: {min_val:.4f}\n'
        stats_text += f'Max: {max_val:.4f}'

        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9,
               family='monospace')

        # Add mean line
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')

        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def update_plots(self):
        """Update all three histograms"""
        # Get data copies with locks
        with self.geometry_lock:
            geo_data = self.geometry_data.copy()

        with self.texture_lock:
            tex_data = self.texture_data.copy()

        with self.observation_score_lock:
            obs_data = self.observation_score_data.copy()

        # Plot geometry complexity
        self.axes[0].clear()
        self.axes[0].set_title('Geometric Complexity')
        self.axes[0].set_xlabel('Complexity Value')
        self.axes[0].set_ylabel('Count')
        self.plot_histogram(self.axes[0], geo_data, bins=30, range_tuple=(0.0, 0.15),
                          color='steelblue')

        # Plot texture complexity
        self.axes[1].clear()
        self.axes[1].set_title('Texture Complexity')
        self.axes[1].set_xlabel('Complexity Value')
        self.axes[1].set_ylabel('Count')
        self.plot_histogram(self.axes[1], tex_data, bins=30, range_tuple=(0.0, 0.2),
                          color='coral')

        # Plot observation score
        self.axes[2].clear()
        self.axes[2].set_title('Observation Score')
        self.axes[2].set_xlabel('Score Value')
        self.axes[2].set_ylabel('Count')
        self.plot_histogram(self.axes[2], obs_data, bins=30, range_tuple=(0.0, 10.0),
                          color='mediumseagreen')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def run(self):
        """Main loop"""
        rate = rospy.Rate(5)  # 5 Hz update rate

        while not rospy.is_shutdown():
            try:
                self.update_plots()
                rate.sleep()
            except KeyboardInterrupt:
                break
            except Exception as e:
                rospy.logerr(f"Error in plot update: {e}")

        plt.close('all')
        rospy.loginfo("Histogram Plotter shutting down.")

if __name__ == '__main__':
    try:
        plotter = HistogramPlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        pass
