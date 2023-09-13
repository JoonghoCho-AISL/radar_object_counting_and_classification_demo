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

# 복소수를 문자열로 변환하는 함수
def complex_to_str(c):
    # 복소수인지 확인
    if isinstance(c, complex):
        return f"{c.real}+{c.imag}j"
    return str(c)

def array_to_list(arr):
    return [[complex_to_str(c) for c in row] for row in arr]

for i in range(300):
    preprocessed_data = sensor.read_data().tolist()
    raw_data = np.array(sensor.raw_data)


    prepro_dict = {prepro_key : preprocessed_data}
    raw_dict = {raw_key : raw_data}

    temp_prepro_path = preprocessed_path + str(i) + '.json'
    temp_raw_path = raw_path + str(i) + '.json'

    save_json(temp_prepro_path, json.dumps(prepro_dict))
    save_json(temp_raw_path, json.dumps(array_to_list(raw_dict)))

    print('save : ', temp_prepro_path)
    print('save : ', temp_raw_path)



    
