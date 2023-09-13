import json
import numpy as np

from radar import radar_prediction

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent = 4)

sensor = radar_prediction('192.168.0.28')

preprocessed_path = 'sample_data/preprocessed_data/preprocessed_data'
raw_path = 'sample_data/raw_data/raw_data'

prepro_key = 'preprocessed_data'
raw_key = 'raw_data'
def complex_encoder(z):
    return {"real": z.real, "imag": z.imag}

# 딕셔너리 (실수와 허수 부분)를 복소수로 변환
def complex_decoder(z_dict):
    return complex(z_dict["real"], z_dict["imag"])

for frame in range(300):
    preprocessed_data = sensor.read_data().tolist()
    raw_data = np.array(sensor.raw_data)
    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            raw_data[i][j] = complex(raw_data[i][j])
    
    raw_data = raw_data.tolist()

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            raw_data[i][j] = complex_encoder(raw_data[i][j])

    print(type(raw_data[0][0]))

    prepro_dict = {prepro_key : preprocessed_data}
    raw_dict = {raw_key : raw_data}

    temp_prepro_path = preprocessed_path + str(frame) + '.json'
    temp_raw_path = raw_path + str(frame) + '.json'

    save_json(temp_prepro_path, json.dumps(prepro_dict))
    save_json(temp_raw_path, json.dumps(raw_dict))

    print('save : ', temp_prepro_path)
    print('save : ', temp_raw_path)



    
