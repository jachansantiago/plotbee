import bqplot as bq
from ipywidgets.widgets import Label, FloatProgress, Button, Image
from ipywidgets.widgets import Layout, HBox, VBox
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display


def hist_widget():
    x_sc = bq.LinearScale(min=0, max=6000)
    y_sc = bq.LinearScale(min=0, max=3)
    x = [0, 0]
    y = [0, 3]
    def_tt = bq.Tooltip(formats=['0.2f'], fields=['midpoint'])
#     panzoom = bq.PanZoom(scales={'x': [x_sc]})
    hist = bq.Hist(sample=[0], scales={'sample': x_sc, 'count': y_sc}, bins=6000, tooltip=def_tt)
    vline = bq.Lines(x=x, y=y, scales={'x': x_sc, 'y': y_sc}, colors=["red"])
    ax_x = bq.Axis(scale=x_sc)
    ax_y = bq.Axis(scale=y_sc, orientation='vertical')

    fig = bq.Figure(marks=[hist, vline], axes=[ax_x, ax_y], padding_y=0, layout=Layout(width="600px", height="400px"))
    return fig, hist, x_sc, vline


def line_widget():
    x_sc = bq.LinearScale()
    y_sc = bq.LinearScale()
    
    x = [0]
    y = [0]
    
    def_tt = bq.Tooltip(formats=['', '0.2f'], fields=['name', 'index'])
#     panzoom = bq.PanZoom(scales={'x': [x_sc]})
    scales = {'x': x_sc, 'y': y_sc}
    x_ax = bq.Axis(scale=x_sc)
    y_ax = bq.Axis(scale=y_sc, orientation="vertical")
    Line = bq.Lines(x=x, y=y, scales=scales, tooltip=def_tt)
    vline = bq.Lines(x=[0,0], y=[0,0], scales={'x': x_sc, 'y': y_sc}, colors=["red"])

    fig = bq.Figure(marks=[Line, vline], axes=[x_ax, y_ax], padding_y=0, layout=Layout(width="600px", height="400px"))
    return fig, Line, x_sc, y_sc, vline