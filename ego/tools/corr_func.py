#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

def add_figure_to_tex (v_file_name, v_title, v_dir, v_now):
    tex_file = open(v_dir + v_file_name + '.txt','w')
    tex_file.write(r'''
\begin{figure}[htbp]
	\centering
	\includegraphics[width=\textwidth]{graphics/pyplots/%s/plots/%s.png}
	\caption{%s}
	\label{img:%s}
\end{figure}
    ''' % (v_now, v_file_name, v_title, v_file_name))
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
