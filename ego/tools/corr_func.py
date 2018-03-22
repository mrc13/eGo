#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corr Functions
"""

# General Packages
import pandas as pd

from matplotlib import pyplot as plt

def corr_district (**kwargs):
    snap_idx = kwargs.get('snap_idx')
    line_df = kwargs.get('line_df')
    mv_line_df = kwargs.get('mv_line_df')
#    trafo_df = kwargs.get('trafo_df')
    mv_trafo_df = kwargs.get('mv_trafo_df')
    level_colors = kwargs.get('level_colors')
#    dist_dir = kwargs.get('dist_dir')
    dist_plot_dir = kwargs.get('dist_plot_dir')
#    dist_corr_dir = kwargs.get('dist_corr_dir')
    make_plot = kwargs.get('make_plot', True)

    columns = [['mv_grid',
                'lev0',
                'lev1',
                'r',
                'lev0_rel_overl_max',
                'lev1_rel_overl_max',
                'cap0',
                'cap1']]
    dist_df = pd.DataFrame(columns=columns)

    mv_trafo_df = mv_trafo_df.set_geometry('grid_buffer')

    for index, row in mv_trafo_df.iterrows():

        mv_grid_id = row['subst_id']
        grid_buffer = row['grid_buffer']


        ## Gets all hv/ehv trafos in this mv grid.
#        mv_grid_hv_trafo_df = trafo_df.loc[
#                trafo_df['geometry'].within(grid_buffer)
#                ]

        ## All relevant voltages in this grid district
        dist_volts = []

        ### HV/MV Trafo voltages
        dist_volts.append(row['v_nom0'])
        dist_volts.append(row['v_nom1'])

        ### HV/EHV and UHV/EHV Trafo voltages
#        dist_volts.extend(mv_grid_hv_trafo_df['v_nom0'].tolist())       # The idea is that a voltage level only counts when there is a trafo connection to it.
#        dist_volts.extend(mv_grid_hv_trafo_df['v_nom1'].tolist())

        dist_volts = set(dist_volts)

        dist_levs = [get_lev_from_volt(volt) for volt in dist_volts]

        ## Select all relevant lines
        ### MV
        dist_mv_lines_df = mv_line_df.loc[mv_line_df['mv_grid'] == mv_grid_id]

        ### HV
        dist_hv_lines_df = line_df.loc[
                [x in dist_volts for x in line_df['v_nom']]
                ]

        dist_hv_lines_df = dist_hv_lines_df.loc[
                dist_hv_lines_df['geometry'].intersects(grid_buffer)
                ]

        ## Calculate grid capacity per level
        columns = dist_levs
        index =   ['s_nom_length_MVAkm']
        dist_cap_df = pd.DataFrame(index=index, columns=columns)

        ### HV
        hv_cap = dist_hv_lines_df.groupby('lev')['s_nom_length_GVAkm'].sum()
        for idx, val in hv_cap.iteritems():
            dist_cap_df.loc['s_nom_length_MVAkm'][idx] = val*1e3
        ### MV
        mv_cap = dist_mv_lines_df.groupby('lev')['s_nom_length_GVAkm'].sum()
        for idx, val in mv_cap.iteritems():
            dist_cap_df.loc['s_nom_length_MVAkm'][idx] = val*1e3

        # Overload Dataframe
        dist_s_sum_len_over_t = pd.DataFrame(0.0,
                                       index=snap_idx,
                                       columns=dist_levs)
        ## Normalized
        dist_s_sum_len_over_t_norm = pd.DataFrame(0.0,
                                       index=snap_idx,
                                       columns=dist_levs)

        for df in [dist_mv_lines_df, dist_hv_lines_df]:

            for i, r in df.iterrows():
                lev = r['lev']

                s_over_series = pd.Series(data=r['s_over_abs'],
                                          index=snap_idx)*1e3       # Then in MVAkm

                dist_s_sum_len_over_t[lev] = (  dist_s_sum_len_over_t[lev]
                                                + s_over_series)

        for col in dist_levs:
            dist_s_sum_len_over_t_norm[col] = (
                    dist_s_sum_len_over_t[col]
                    / dist_cap_df.loc['s_nom_length_MVAkm'][col]
                    * 100)


        ## Cleaning out levels without overload
#        for column in dist_s_sum_len_over_t_norm.columns:
#            max_over = dist_s_sum_len_over_t_norm[column].max()
#            if max_over < 0.8:
#                dist_s_sum_len_over_t_norm = dist_s_sum_len_over_t_norm.drop([column], axis=1)
#                dist_s_sum_len_over_t = dist_s_sum_len_over_t.drop([column], axis=1)
#
#        if dist_s_sum_len_over_t.empty:
#            continue

        if make_plot == True:
            plt_name = "Relative district overloading"
            fig, ax1 = plt.subplots(2, sharex=True) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
            fig.set_size_inches(10,6)


            dist_s_sum_len_over_t_norm.plot(
                    kind='line',
                    title=plt_name,
                    legend=True,
                    color=[level_colors[lev] for lev in  dist_s_sum_len_over_t.columns],
                    linewidth=2,
                    ax = ax1[0])
            mvhv_p = pd.Series(data=row['p'], index=snap_idx)
            mvhv_p.plot(
                    kind='line',
                    ax = ax1[1])

            file_name = 'district_overloading_mv_grid_' + str(mv_grid_id)
            fig.savefig(dist_plot_dir + file_name + '.pdf')
            fig.savefig(dist_plot_dir + file_name + '.png')
            plt.close(fig)

            plt_name = "Grid District"
            fig, ax1 = plt.subplots(1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
            fig.set_size_inches(6,6)
            xmin, ymin, xmax, ymax = grid_buffer.bounds

            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])

            mv_trafo_df = mv_trafo_df.set_geometry('grid_buffer')
            mv_trafo_df[mv_trafo_df['subst_id'] == mv_grid_id].plot(ax=ax1,
                       alpha = 0.3,
                       color = 'y'
                       )
            mv_trafo_df = mv_trafo_df.set_geometry('geometry')
            mv_trafo_df[mv_trafo_df['subst_id'] == mv_grid_id].plot(ax=ax1,
                       alpha = 1,
                       color = 'r',
                       marker='o',
                       markersize=300,
                       facecolors='none'
                       )

            ax1 = add_plot_lines_to_ax(dist_hv_lines_df, ax1, level_colors, 3)
            ax1 = add_plot_lines_to_ax(dist_mv_lines_df, ax1, level_colors, 1)


            file_name = 'district_' + str(mv_grid_id)
            fig.savefig(dist_plot_dir + file_name + '.pdf')
            fig.savefig(dist_plot_dir + file_name + '.png')
            plt.close(fig)

        corr_df = dist_s_sum_len_over_t.corr()
        for idx, row in corr_df.iterrows():
            for col in corr_df.columns:
                if idx==col:
                    continue
                r =  corr_df.loc[idx][col]
                lev0_rel_overl_max = dist_s_sum_len_over_t_norm[idx].max()
                lev1_rel_overl_max = dist_s_sum_len_over_t_norm[col].max()
                dist_df = dist_df.append({'mv_grid': mv_grid_id,
                                'lev0': idx,
                                'lev1': col,
                                'r': r,
                                'lev0_rel_overl_max': lev0_rel_overl_max,
                                'lev1_rel_overl_max': lev1_rel_overl_max,
                                'cap0': dist_cap_df.loc['s_nom_length_MVAkm'][idx],
                                'cap1': dist_cap_df.loc['s_nom_length_MVAkm'][col]},
                            ignore_index=True)

    return dist_df

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
