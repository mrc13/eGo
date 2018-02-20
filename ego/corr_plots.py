# -*- coding: utf-8 -*-
"""
This is the post processing file for Malte's Master thesis 

"""

import pandas as pd
import os

from etrago.appl import etrago
from ego.tools.plots import (make_all_plots,plot_line_loading, plot_stacked_gen,
                                 add_coordinates, curtailment, gen_dist,
                                 storage_distribution, igeoplot)
# For importing geopandas you need to install spatialindex on your system http://github.com/libspatialindex/libspatialindex/wiki/1.-Getting-Started
from ego.tools.utilities import get_scenario_setting, get_time_steps
from ego.tools.io import geolocation_buses, etrago_from_oedb
from ego.tools.results import total_storage_charges
from sqlalchemy.orm import sessionmaker
from egoio.tools import db
from etrago.tools.io import results_to_oedb, NetworkScenario

# Procedure

args = get_scenario_setting(json_file='scenario_setting.json')

## Scenario Query
#args = args['eTraGo']
#scenario = NetworkScenario(session,
#                           version=args['gridversion'],
#                           prefix=args['ormcls_prefix'],
#                           method=args['method'],
#                           start_snapshot=args['start_snapshot'],
#                           end_snapshot=args['end_snapshot'],
#                           scn_name=args['scn_name'])
#
#scenario.fetch_by_relname(name = 'Load')
#scenario.fetch_by_relname(name = 'Storage')
#scenario.fetch_by_relname(name = 'Transformer')
#scenario.fetch_by_relname(name = 'Line')
#scenario.fetch_by_relname(name = 'Bus')
#scenario.fetch_by_relname(name = 'Generator')


# eTraGo Results processing

# cheats
## Buses
eTraGo.buses.at[str(16995),'v_nom']
eTraGo.buses_t.p['16995']
eTraGo.buses[eTraGo.buses['bus'] == 380.0]

eTraGo.lines[eTraGo.lines['bus0'] == '16995']
eTraGo.lines

eTraGo.lines.loc['13722'] # loc ist für den Index

bus_id = 25740
eTraGo.generators[eTraGo.generators['bus'] == str(bus_id)]
eTraGo.generators.loc['9541']
eTraGo.generators_t.p['9541']

eTraGo.generators.groupby("carrier")["p_nom"].sum()

plot_line_loading(eTraGo)

eTraGo.storage_units




# My plots
from matplotlib import pyplot as plt
import numpy as np

plot_dir = '/home/student/Dropbox/Masterarbeit/Thesis/graphics/pyplots/'

## eTraGo grid Plot (with PyPSA plot function)
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,12)

# eTraGo.plot(bus_sizes=10)

load_distribution = eTraGo.loads_t.p_set.loc[eTraGo.snapshots[0]].groupby(eTraGo.loads.bus).sum()

eTraGo.plot(bus_sizes=6*load_distribution,ax=ax1,title="Load distribution")

fig.savefig(plot_dir + 'eTraGo_grid.pdf')
fig.savefig(plot_dir + 'eTraGo_grid.png')


## eTraGo Line Plot
lines = eTraGo.lines
lines_t = eTraGo.lines_t.p0

fig, ax = plt.subplots(4, sharex=True, figsize=(40, 60))

fig.set_size_inches(13,8)

l_id = 7231
l_s_nom = lines.at[str(l_id),'s_nom_opt']
l_t_p0 = lines_t[str(l_id)]
ax[0].plot(l_t_p0.index, abs(l_t_p0), color = 'blue', linewidth=4) 

l_id = 23790
l_s_nom = lines.at[str(l_id),'s_nom_opt']
l_t_p0 = lines_t[str(l_id)]
ax[1].plot(l_t_p0.index, abs(l_t_p0), color = 'green', linewidth=4) 

l_id = 9695
l_s_nom = lines.at[str(l_id),'s_nom_opt']
l_t_p0 = lines_t[str(l_id)]
ax[2].plot(l_t_p0.index, abs(l_t_p0), color = 'green', linewidth=4) 

gen_p = eTraGo.generators_t.p['9541']
ax[3].plot(gen_p.index, gen_p, color = 'grey', linewidth=4) 


fig.savefig(plot_dir + 'lines_examp.pdf')
fig.savefig(plot_dir + 'lines_examp.png')


## Folium Plots
igeoplot(network=eTraGo, session=session, args=args)  


## Plotly Plots


# pltly.init_notebook_mode(connected=True)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot

network = eTraGo

tech = '13' # in ["Gas","Brown Coal","Hard Coal","Wind Offshore","Wind Onshore","Solar"]

gens = network.generators[network.generators.carrier == tech]
gen_distribution = gens.groupby("bus").sum()["p_nom"].reindex(network.buses.index,fill_value=0.)

#set the figure size first
fig = dict(data=[],layout=dict(width=700,height=700))

fig = network.iplot(bus_sizes=0.05*gen_distribution, fig=fig,
                     bus_text=tech + ' at bus ' + network.buses.index + ': ' + round(gen_distribution).values.astype(str) + ' MW',
                     title=tech + " distribution")   

plot(fig)





## Cheat section


from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch

fig = plt.figure(1, figsize=(5,5), dpi=90)
ring_mixed = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 0.8), (0, 0)])
ax = fig.add_subplot(111)
ring_patch = PolygonPatch(ring_mixed)
ax.add_patch(ring_patch)

ax.set_title('Filled Polygon')
xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
#ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
#ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)


import plotly.plotly as py
import plotly.figure_factory as ff
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/school_earnings.csv")

table = ff.create_table(df)
py.iplot(table, filename='table1')



import numpy as np
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
print (__version__) # requires version >= 1.9.0

#Always run this the command before at the start of notebook
init_notebook_mode(connected=True)
import plotly.graph_objs as go

x=np.array([2,5,8,0,2,-8,4,3,1])
y=np.array([2,5,8,0,2,-8,4,3,1])


data = [go.Scatter(x=x,y=y)]
fig = go.Figure(data = data,layout = go.Layout(title='Offline Plotly Testing',width = 800,height = 500,
                                           xaxis = dict(title = 'X-axis'), yaxis = dict(title = 'Y-axis')))


plot(fig,show_link = False)




import plotly
plotly.tools.set_credentials_file(username='maltesc', api_key='x7rbNYEtKtOzV5L1tnFy')






import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

y0 = np.random.randn(50)-1
y1 = np.random.randn(50)+1

trace0 = go.Box(
    y=y0
)
trace1 = go.Box(
    y=y1
)
data = [trace0, trace1]
py.iplot(data)