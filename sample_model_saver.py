from real_time_prediction import ml_model

ml = ml_model()
model = ml.ml_model

model.save('basemodel.h5')
