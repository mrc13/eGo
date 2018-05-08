#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six

"""
Corr Functions
"""

def corr_colors (v_corr=0):
    try:
        i = round(abs(v_corr), 2)
        rgb = (1-i,i,0)
        return rgb
    except:
        return (0.5, 0.5, 0.5)


def color_for_s_over(i):
    if i > 1.0:
        i = 1.0

    rgba = (i,0,0,i)
    return rgba

def to_str (v):
    if isinstance(v, str):
        return v

    if isinstance(v, int):
        return v
    try:
        i = '%.2f' % v
        return i
    except:
        return v

def get_lev_from_volt (
        v_voltage,
        v_aggr = False): # in kV
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
        if v_aggr==True:
            return 'EHV'
        else:
            return 'EHV220'
    elif v == 380:
        if v_aggr==True:
            return 'EHV'
        else:
            return 'EHV380'
    else: return 'unknown'


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


def add_plot_lines_to_ax(v_line_df,
                         v_ax,
                         v_level_colors,
                         v_size=1):
    levs = set(v_line_df['lev'])
    for lev in levs:
        plot_df = v_line_df.loc[v_line_df['lev'] == lev]
        plot_df.plot(
                color=v_level_colors[lev],
                linewidth=v_size,
                ax = v_ax)

    return v_ax

def add_weighted_plot_lines_to_ax(v_line_df,
                         v_ax,
                         v_size=1):

    plot_df = v_line_df

    plot_df.plot(
            color=plot_df['color'],
            linewidth=v_size,
            ax = v_ax)

    return v_ax

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
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
    return fig, ax


def render_corr_table(data, col_width=3.0, row_height=0.625, font_size=12,
                     header_color='grey', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, first_width=5.0,
                     ax=None, **kwargs):
    if ax is None:

        index_name = data.index.name
        data = data.reset_index()
        data = data.rename(columns={index_name: ''})

        size = (np.array(data.shape[::-1]) + np.array([1, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    col_widths = [col_width for x in data.columns]
    col_widths[0] = first_width

    colors = data.applymap(lambda x: corr_colors(x))
    data = data.applymap(lambda x: to_str(x))

    mpl_table = ax.table(cellText=data.values,
                         cellColours=colors.as_matrix(),
                         bbox=bbox,
                         colLabels=data.columns,
                         colWidths=col_widths,
                         colLoc='center')

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        if k[1] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)

    return fig, ax

def corr(X, Y):
    """Computes the Pearson correlation coefficient and a 95% confidence
    interval based on the data in X and Y."""

    r = np.corrcoef(X, Y)[0,1]
    f = 0.5*np.log((1+r)/(1-r))
    se = 1/np.sqrt(len(X)-3)
    ucl = f + 2*se
    lcl = f - 2*se

    lcl = (np.exp(2*lcl) - 1) / (np.exp(2*lcl) + 1)
    ucl = (np.exp(2*ucl) - 1) / (np.exp(2*ucl) + 1)

    return r,lcl,ucl