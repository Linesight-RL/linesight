import argparse
import pandas
import os
import math
import time
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Parameters
Target_Times = [53.76] #Racing times at which to draw a horizontal line
Duration = 20 #seconds
Y_Axis_Percentile = 80 #Only show up to this percentile of values
#Axis_Hour_Range = 0.2 #Percentage of total training time
DPI = 200
FPS = 60
Scatter_Size = 60
Scatter_Opacity = 0.25
Target_Resolution = np.array([2560,1440])
Y_Axis_Margin = 0.05 #Percentage
Label_Fontsize = 30
Ticks_Fontsize = 18
Annotate_Fontsize = 18

#Calculated from parameters
Total_Intervals = math.ceil(Duration*FPS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", type=str, required=True)
    args = parser.parse_args()
    df = pandas.read_csv(args.inputs)
    df['Wall time'] -= df['Wall time'].iloc[0]
    df['Wall time'] /= 60*60 #seconds to hours
    df['Race time min'] = np.minimum.accumulate(df['Value'])

    fig, ax = plt.subplots(figsize=Target_Resolution/DPI)
    x_min = df['Wall time'].min()
    x_max = df['Wall time'].max()
    twin_ax = ax.twinx()
    
    def animate(i):
        points_to_show = round((i+1)*len(df['Wall time'])/Total_Intervals)
        print(f"Plotting {points_to_show}/{len(df['Value'])}")
        ax.clear()
        #axis_relevant_values = df['Value'].iloc[max(0,round(points_to_show-len(df['Value'])*Axis_Hour_Range)):points_to_show]
        #y_max = axis_relevant_values.max()
        #y_min = axis_relevant_values.min()
        axis_relevant_values = df['Value'].iloc[:points_to_show]
        y_min = axis_relevant_values.min()
        y_max = np.percentile(axis_relevant_values, Y_Axis_Percentile)
        ax.tick_params(labelsize=Ticks_Fontsize)
        twin_ax.tick_params(labelsize=Ticks_Fontsize)
        twin_ax.set_yticks(Target_Times)
        ax.set_ylim(y_min-(y_max-y_min)*Y_Axis_Margin,y_max)
        twin_ax.set_ylim(ax.get_ylim())
        ax.set_xlim(x_min,x_max)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        ax.set_xlabel('Training hours', fontsize=Label_Fontsize)
        ax.set_ylabel('Race time', fontsize=Label_Fontsize)
        for t in Target_Times:
            ax.hlines(t,x_min,x_max, linestyle = ((0, (1, 4))), color='green')
        ax.scatter(df['Wall time'].iloc[:points_to_show], df['Value'].iloc[:points_to_show], s=Scatter_Size, color='blue', alpha=Scatter_Opacity)#, linewidth=0
        ax.plot(df['Wall time'].iloc[:points_to_show], df['Race time min'].iloc[:points_to_show], color='red')
        ax.annotate("{:.2f}".format(round(df['Race time min'].iloc[points_to_show-1],2)),xy=(df['Wall time'].iloc[points_to_show-1]-0.05*(x_max-x_min),df['Race time min'].iloc[points_to_show-1]-(y_max-y_min)*Y_Axis_Margin*0.75),color='red',fontsize=Annotate_Fontsize)
    ani = animation.FuncAnimation(fig,animate,frames=Total_Intervals)
    FFwriter = animation.FFMpegWriter(fps=FPS)
    Start_Time = time.perf_counter()
    ani.save('animated_race_time.mp4', writer = FFwriter, dpi=DPI)
    print("Took",time.perf_counter()-Start_Time,"s to make mp4")
