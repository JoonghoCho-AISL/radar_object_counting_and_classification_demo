from tkinter import S
import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time
import json
import os
import argparse
import tensorflow as tf
from models import basemodel_rasp

def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

class radar_prediction():
    def __init__(self,
                    ip_address='127.0.0.1',
                    start_point = 160,
                    num_points = 301,
                    hwaas = 8,
                    sweep_per_frame = 10,
                    frame_rate = 10
                    ):
        self.client = a121.Client(ip_address = ip_address)

        self.client.connect()
        #start_distance = start_point * 2.5mm
        start_point = 160 # 400mm
        #end_distance = start_point * 2.5mm + num_points * 2.5mm 
        # num_points = (end_dis - start_point * 2.5) / 2.5 + 1
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

        self.client.setup_session(sensor_config)
        self.client.start_session()
    
    def read_data(self):
        raw_data = self.client.get_next()
        frame = raw_data.frame
        data = np.expand_dims(np.abs(np.mean(frame, axis=0)), axis = 0)
        return data

        
    
    def disconnect(self):
        self.client.disconnect()

class ml_model():
    def __init__(self, path = 'model_weights/basemodel', input_shape = (1, 301)):
        self.label_dict= read_json('label.json')
        self.idx2label = {v:k for k, v in self.label_dict.items()}
        self.ml_model = basemodel_rasp(len(self.label_dict))
        self.ml_model.load_weights(path).expect_partial()
        self.ml_model.build(input_shape = input_shape)
    
    def predictions(self):
        self.inference = ml_model.predict(self.data)

    def get_inference(self, data):
        self.data = data
        self.predictions()
        return self.inference

    def get_label(self, data):
        self.data = data
        self.predictions()
        label = np.argmax(self.inference)
        inf = self.idx2label[label[0]]
        return inf
        

def main():
    ip = '192.168.0.28'
    radar = radar_prediction(ip)
    ml = ml_model()
    for i in range(200):
        data = radar.get_data()
        inf = ml.get_label(data)
        print(inf)

if __name__ == "__main__":
    main()
