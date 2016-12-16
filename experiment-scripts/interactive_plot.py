"""Create interactive plot from xy coordinates of TSNE embeding"""

import numpy as np
import matplotlib.pyplot as plt
from bokeh.models import HoverTool, TapTool, CustomJS
from bokeh.plotting import figure, show, save, output_file, ColumnDataSource
from matplotlib.colors import rgb2hex


def plot_2D_scatter(xx, yy, labels=None, df=None, html_file='temp.html'):
    '''Plot a 2D scatter plot and add functionality on mouse hover and mouse click
    
    Parameters
    ----------
    xx : np.array
        The x coordinates of the data points to be plotted.
    yy : np.array
        The y coordinates of the data points to be plotted.
    labels : np.array
        Labels for each data point control the color of the plotted data.
    df : pandas.DataFrame
        Associated metadata to be displayed with interactive functionality,
        use tags of country, culture, language, genre, and url, if available.
    
    Returns
    -------
    p : bokeh.plot
        Interactive plot in html format.
    '''
    label_colors = []
    if labels is not None:
        classes = np.unique(labels)
        colors = plt.cm.spectral(np.linspace(0, 1, len(classes)))
        bokehcolors = [rgb2hex(cc) for cc in colors]
        label_colors = [bokehcolors[np.where(classes==classlab)[0][0]] for 
                        classlab in labels]

    countries = []
    info = []
    urls = []
    if df is not None:
        for ind in df.index:
            countries.append(df['Country'].iloc[ind])
            info.append([df['Culture'].iloc[ind], df['Language'].iloc[ind], 
                         df['Genre'].iloc[ind]])
            urls.append(df['Url'].iloc[ind])

    source = ColumnDataSource(data=dict(
        x = list(xx),
        y = list(yy),
        color = label_colors,
        name = countries,
        info = info,
        url = urls
    ))
    
    TOOLS = "pan, wheel_zoom, box_zoom, reset, save"
    
    p = figure(tools=TOOLS)
    r1 = p.cross('x', 'y', fill_color='color', size=10, alpha=0.7, 
                 line_color=None, source=source)
    markers = ['asterisk', 'circle', 'cross', 'diamond', 'circle_x', 
               'inverted_triangle', 'square', 'diamond_cross', 
               'square_x', 'triangle']    
    if labels is not None:
        for label in np.unique(labels):
            inds = np.where(labels==label)[0]
            eval('p.'+markers[label % len(markers)]+'(np.array(xx)[inds], np.array(yy)[inds], fill_color=np.array(label_colors)[inds], size=10, alpha=0.7, line_color=np.array(label_colors)[inds], legend="Cluster " + str(label+1))')
        
    hover_tooltips = """
        <div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@name</span>
            </div>
            <div>
                <span style="font-size: 12px;">@info</span>
            </div>
        </div>
        """

    callback = CustomJS(args=dict(r1=r1), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        var d1 = cb_obj.get('data');
        url = d1['url'][inds[0]];
        if (url){
            window.open(url);
        }
        """)

    p.add_tools(HoverTool(renderers=[r1], tooltips=hover_tooltips))
    p.add_tools(TapTool(renderers=[r1], callback = callback))
    
    output_file(html_file)
    save(p)
    show(p)