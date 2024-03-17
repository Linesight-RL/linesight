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
Target_Times = [] #Racing times at which to draw a horizontal line
Target_Times_Labels = [] #Leave empty for default labels (time)
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
Draw_Points = True
Draw_Min = True
Draw_References = True

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

    plt.rcParams.update({
        "figure.facecolor":  (1.0, 1.0, 1.0, 0.),
        "axes.facecolor":    (1.0, 1.0, 1.0, 0.),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.),
    })
    x_min = df['Wall time'].min()
    x_max = df['Wall time'].max()
    fig, ax = plt.subplots(figsize=Target_Resolution/DPI)
    twin_ax = ax.twinx()
    fig.patch.set_alpha(0.0)
    #ax.patch.set_alpha(0.0)
    ax.set_xlabel('Training hours', fontsize=Label_Fontsize)
    ax.set_ylabel('Race time', fontsize=Label_Fontsize)
    ax.tick_params(labelsize=Ticks_Fontsize)
    twin_ax.tick_params(labelsize=Ticks_Fontsize)

    if Draw_References:
        reference_lines = ax.hlines(Target_Times,0,df['Wall time'].max(), linestyle = ((0, (1, 4))), color='green')
    if Draw_Points:
        points_scatter = ax.scatter([], [], s=Scatter_Size, color='blue', alpha=Scatter_Opacity)#, linewidth=0
    if Draw_Min:
        min_line = ax.plot([], [], color='red')[0]
        min_text = ax.annotate("",xy=(df['Wall time'].max()/2,df['Value'].mean()),color='red',fontsize=Annotate_Fontsize)
    
    def animate(i):
        points_to_show = max(1,round((i+1)*len(df['Wall time'])/Total_Intervals))
        print(f"Plotting {points_to_show}/{len(df['Value'])}")
        #ax.clear()
        #ax.patch.set_visible(False) 
        #axis_relevant_values = df['Value'].iloc[max(0,round(points_to_show-len(df['Value'])*Axis_Hour_Range)):points_to_show]
        #y_max = axis_relevant_values.max()
        #y_min = axis_relevant_values.min()
        axis_relevant_values = df['Value'].iloc[:points_to_show]
        y_min = axis_relevant_values.min()
        y_max = np.percentile(axis_relevant_values, Y_Axis_Percentile)
        twin_ax.set_yticks(Target_Times,Target_Times_Labels if len(Target_Times_Labels)>0 else None)
        ax.set_ylim(y_min-(y_max-y_min)*Y_Axis_Margin,y_max)
        twin_ax.set_ylim(ax.get_ylim())
        ax.set_xlim(x_min,x_max)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        if Draw_References:
            #reference_lines.set_offsets((x_min,x_max))
            pass
        if Draw_Points:
            points_scatter.set_offsets(df[['Wall time','Value']].iloc[:points_to_show])
        if Draw_Min:
            min_line.set_data(df['Wall time'].iloc[:points_to_show], df['Race time min'].iloc[:points_to_show])
            min_text.set_text("{:.2f}".format(round(df['Race time min'].iloc[points_to_show-1],2)))
            min_text.set_position((df['Wall time'].iloc[points_to_show-1]-0.05*(x_max-x_min),df['Race time min'].iloc[points_to_show-1]-(y_max-y_min)*Y_Axis_Margin*0.75))
    ani = animation.FuncAnimation(fig,animate,frames=Total_Intervals)
    FFwriter = animation.FFMpegWriter(fps=FPS, codec="png") #, extra_args=['-pix_fmt', 'yuva444p']
    #FFwriter = animation.PillowWriter(fps=FPS, codec="gif") #,extra_args=['-pix_fmt', 'yuva444p']
    #FFwriter = animation.ImageMagickWriter(fps=FPS, codec="gif")
    Start_Time = time.perf_counter()
    ani.save('animated_race_time.mov', writer = FFwriter, dpi=DPI) #,savefig_kwargs={"transparent": True}
    print("Took",time.perf_counter()-Start_Time,"s to make mp4")
