from tkinter import S
import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time
from tqdm import tqdm
import os
from json import dumps, loads
# from post import posting
# from pub_sub import publishing
# from sub_sub import subscribing
from multiprocessing import Process
from multiprocessing import Pool
import argparse
import tensorflow as tf

client = a121.Client(ip_address = '192.168.0.165')
# client.ip_address = '127.0.0.1'
client.connect()
#start_distance = start_point * 2.5mm
start_point = 160 # 400mm
#end_distance = start_point * 2.5mm + num_points * 2.5mm
# num_points = (end_dis - start_point * 2.5) / 2.5
num_points = 301 # 1,150mm

sensor_config = a121.SensorConfig(
    subsweeps=[
        a121.SubsweepConfig(
            start_point = start_point,
            step_length = 1,
            num_points = num_points,
            profile = a121.Profile.PROFILE_1,
            hwaas = 8,
        ),
    ],
    sweeps_per_frame = 10,
    frame_rate = 10,
)

client.setup_session(sensor_config)
client.start_session()