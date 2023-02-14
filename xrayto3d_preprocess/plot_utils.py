import matplotlib
import matplotlib.pyplot as plt
from typing import Dict

matplotlib.style.use('ggplot')
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['lines.markersize'] = 10

def get_barplot(data:Dict):

    fig,ax = plt.subplots()
    labels = list(data.keys())
    count = list(data.values())

    # creating the bar plot
    plt.bar(labels, count, width=0.4)
    plt.xticks(fontsize=10)
    
    return fig,ax