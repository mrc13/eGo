#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six

"""
Corr Functions
"""

def get_lev_from_volt (v_voltage): # in kV
    try:
        v = float(v_voltage)
    except:
        return None
    if v <= 1:
        return 'LV'
    elif (v >= 3) & (v <= 30):
        return 'MV'
    elif (v >= 60) & (v <= 110):
        return 'HV'
    elif v == 220:
        return 'EHV220'
    elif v == 380:
        return 'EHV380'
    else: return 'unknown'

def get_volt_from_lev (v_lev):
    if v_lev == 'MV':
        return 20. # This is not always true
    elif v_lev == 'HV':
        return 110.
    elif v_lev == 'EHV220':
        return 220.
    elif v_lev == 'EHV380':
        return 380.
    else: return None

def get_hour_of_year (v_d):
    return ((v_d.timetuple().tm_yday-1) * 24 + v_d.hour + 1)

def add_figure_to_tex (v_title, v_dir, v_file_name):
    tex_file = open(v_dir + v_file_name + '.tex','w')
    tex_file.write(r'''
\begin{figure}[htbp]
	\centering
	\includegraphics[width=\textwidth]{graphics/pyplots/%s%s.png}
	\caption{%s}
	\label{img:%s}
\end{figure}
    ''' % (v_dir, v_file_name, v_title, v_file_name))
    tex_file.close()

def add_table_to_tex (v_title, v_dir, v_file_name):
    tex_file = open(v_dir + v_file_name + '.tex','w')
    tex_file.write(r'''
\begin{table}[htbp]
    \caption{%s}
	\centering
	\includegraphics[width=\textwidth]{graphics/pyplots/%s%s.png}
	\label{img:%s}
\end{table}
    ''' % (v_title, v_dir, v_file_name, v_file_name))
    tex_file.close()




def add_plot_lines_to_ax(v_line_df, v_ax, v_level_colors, v_size):
    levs = set(v_line_df['lev'])
    for lev in levs:
        plot_df = v_line_df.loc[v_line_df['lev'] == lev]
        plot_df.plot(
                color=v_level_colors[lev],
                linewidth=v_size,
                ax = v_ax)

    return v_ax

#data = hvmv_comparison_df
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=12,
                     header_color='grey', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, first_width=5.0,
                     ax=None, **kwargs):
    if ax is None:
        data = data.reset_index()
        data = data.rename(columns={'index': ''})
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    col_widths = [col_width for x in data.columns]
    col_widths[0] = first_width
    mpl_table = ax.table(cellText=data.values,
                         bbox=bbox,
                         colLabels=data.columns,
                         colWidths=col_widths)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        if k[1] == 0:
            cell.set_text_props(weight='bold', color='grey')
            cell.set_facecolor('w')
    return fig, ax
