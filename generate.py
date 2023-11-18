
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Final, List, Tuple, Union, Optional

import time
# from PIL import Image

import pyntcloud
import pandas as pd
import subprocess

import plotly.express as px
import plotly.graph_objs as go
import random
import tqdm
import numpy as np


from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader


from av2.geometry.se3 import SE3
import av2.structures
from av2.utils.typing import NDArrayByte, NDArrayInt, NDArrayFloat, NDArrayBool

import sys
import yaml
import argparse

def load_lidar(**kwargs):
    print(dict(
        data_dir=Path(kwargs['data_dest']), labels_dir=Path(kwargs['data_dest'])
        ))
    loader = AV2SensorDataLoader(data_dir=Path(kwargs['data_dest']), labels_dir=Path(kwargs['data_dest']))
    lidar_timestamps = loader.get_ordered_log_lidar_timestamps(kwargs['log_id'])
    if len(lidar_timestamps) == 0 or kwargs['force_redownload']:
        print("Downloading...")
        subprocess.run(
            f's5cmd --no-sign-request cp "s3://argoverse/datasets/av2/lidar/train/{kwargs["log_id"]}/*" {kwargs["data_dest"]}/{kwargs["log_id"]} > /dev/null',
            shell=True
        )
        print("Finished!")
    return loader

#frames
def generate_frame_xyzs(loader, **kwargs):
    lidar_timestamps = loader.get_ordered_log_lidar_timestamps(kwargs['log_id'])
    compensate = not kwargs['no_compensate']
    
    offset = None
    to_render_frames = []
    for i, ts in tqdm.tqdm(list(enumerate(lidar_timestamps[:])), "Reading"):
        sweeppath = loader.get_lidar_fpath_at_lidar_timestamp(kwargs['log_id'],ts)
        sweep = av2.structures.sweep.Sweep.from_feather(sweeppath)
        
        num_points = sweep.xyz.shape[0]
        rows_id = random.sample(range(0,num_points), num_points)[:kwargs['points_per_frame']]
        to_render = sweep.xyz[rows_id]
    
        scalar = sweep.intensity
        scalar = np.linalg.norm(sweep.xyz[rows_id], axis=1).reshape(-1,1)
        
        if compensate:
            transformer = loader.get_city_SE3_ego(kwargs['log_id'], ts)
            to_render = transformer.transform_point_cloud(sweep.xyz)[rows_id]
    
            if offset is None:
                offset = np.mean(to_render, axis = 0)
            to_render = to_render - offset
    
        with_intensity = np.concatenate([to_render, scalar], axis = 1)
        to_render_frames.append(with_intensity)
    return to_render_frames

def generate_plys(to_render_frames, **kwargs):
    subprocess.run(
        f'mkdir -p {all_args["out_dir"]}/plys/{all_args["log_id"]}',
        shell=True
    )
    
    for i,f in tqdm.tqdm(list(enumerate(to_render_frames)), "Writing Plys"):
        outf = np.copy(f)
        cloud = pyntcloud.PyntCloud(pd.DataFrame(outf, columns=['x','y','z','f']))
        cloud.to_file(f'{kwargs["out_dir"]}/plys/{kwargs["log_id"]}/{i:06}.ply')

def generate_preview(to_render_frames, **kwargs):
    subprocess.run(
        f'mkdir -p {kwargs["out_dir"]}/previews/{kwargs["log_id"]}',
        shell=True
    )
    
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "frame:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": None,
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    frames = []
    for i in tqdm.tqdm(range(len(to_render_frames)), "Rendering Preview"):
        to_render = to_render_frames[i]
        downsample = random.sample(
            range(len(to_render)), 
            kwargs['preview_downsample']
        )

        to_render = to_render[downsample]
        scalar = to_render[:, 3] ** 0.5
        
        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=to_render[:, 0],
                    y=to_render[:, 1],
                    z=to_render[:, 2],
                    hovertemplate=None,
                    hoverinfo='skip',
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=scalar,  # set color to an array/list of desired values
                        colorscale='turbo_r',  # choose a colorscale
                        opacity=0.8,
                    ),
                )],                   
            traces= [0],
            name=i,
            layout = dict(transition = dict(duration=100,))
        ) )

        slider_step = {"args": [
            [i],
            {"frame": {"duration": 100, "redraw": True},
                "mode": "immediate",
                "transition": None}
        ],
            "label": i,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)



    fig = px.scatter_3d()

    axis_config = dict(
        backgroundcolor='rgb(0, 0, 0)',
        gridcolor='gray',
        showgrid=False,
        showline=False,
        showticklabels=False,
        showbackground=True,
        zerolinecolor='gray',
        tickfont=dict(color='gray'),
        range=[-256,256],
    )
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": True},
                                    "mode": 'immediate',
                                    "fromcurrent": True, "transition": {"duration": 50, }}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 50}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-0.10, y=-0.05, z=0.10)
    )

    fig.update_layout(
        showlegend=False,
        scene=dict(xaxis=axis_config, yaxis=axis_config, zaxis=axis_config),
        width=1000,
        height=750,
        paper_bgcolor='black',
        plot_bgcolor='rgba(0,0,0,0)',
        scene_aspectmode='cube',
        scene_camera=camera,
        updatemenus=updatemenus,
        hovermode=False,
        sliders = [sliders_dict])
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1,
    )
    fig.update_scenes(xaxis_showspikes=False)
    fig.update_scenes(yaxis_showspikes=False)
    fig.update_scenes(zaxis_showspikes=False)
    fig.update(frames=frames)

    fig.write_html(f'{kwargs["out_dir"]}/previews/{kwargs["log_id"]}.html')
    del fig

'''
# ARGPARSE #
'''
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--args',
                    action='store',
                    help='path to argument file yaml',
                    required=True
                   )
parser.add_argument('-l', '--log-id',
                    dest = 'log_ids',
                    action='append',
                    help='Potentially multiple log IDs to load and generate',
                    required=True
                   )

parser.add_argument('-p', '--previews',
                    action='store_true',
                    help='Generate html previews',
                   )
parser.add_argument('-f', '--ply-frames',
                    action='store_true',
                    help='Generate ply frames',
                   )

parser.add_argument('--force-redownload',
                    action = 'store_true',
                    help='Force redownload of point clouds. '
                        'The script does basic checks for whether the data has already been downloaded, '
                        'but if files are accidentally deleted you may encouter an error.'
                   )
parser.add_argument('--no-compensate',
                    action = 'store_true',
                    help='Generate preview and pointclouds in car reference frame instead of global'
                   )

parser.add_argument('--decompose-sweep',
                    action = 'store',
                    help='generate N frames within a sweep based on timestamp of beams',
                    type = int,
                    metavar='N',
                    default=1
                   )

parser.add_argument('--start',
                    action = 'store',
                    help='Optional start frame',
                    type=int,
                    default=0
                   )

parser.add_argument('--end',
                    action = 'store',
                    help='Optional end frame',
                    type=int,
                    default=999
                   )


parsed = parser.parse_args(sys.argv[1:])
with open(parsed.args) as f:
    yaml_args = yaml.safe_load(f)
yaml_args.update(vars(parsed))
all_args = yaml_args
print(yaml.safe_dump(all_args))

for l in all_args['log_ids']:
    all_args['log_id'] = l
    loader = load_lidar(**all_args)
    frames = generate_frame_xyzs(loader, **all_args)

    #ply
    generate_plys(frames, **all_args)
    if all_args['previews']:
        generate_preview(frames, **all_args)


# #preview



