import os
import numpy as np
import onnx
import time
import onnxruntime as ort
#-------------以下是数据要求-----------
'''
surface: 4 surface variables (MSLP, U10, V10, T2M in the exact order)
upper:(Z, Q, T, U and V) 13levs
(1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa,
 250hPa, 200hPa, 150hPa, 100hPa and 50hPa
721*1440 latitude longitude
initial fields of at 12:00UTC, 2018/09/27.
Note that ndarray (.astype(np.float32)), not in double precision.
'''
# The directory of your input and output data
start_time=time.time()
epoch_start_time=time.time()
input_data_dir1 =r"surface_data"
input_data_dir2 =r"upper_data"
output_data_dir = r"output_data"
model_24 = onnx.load("model\pangu_weather_24.onnx")
input_upper = np.load(os.path.join(input_data_dir2,f'z-q-t-u-v-2020-09-11T21.npy'))
input_surface=np.load(os.path.join(input_data_dir1,f'mslp-10mU-10mV-2mT-2020-09-11T21.npy'))
# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 1
 
# Set the behavier of cuda provider
providers=['CUDAExecutionProvider']
 
# Initialize onnxruntime session for Pangu-Weather Models
ort_session_24 = ort.InferenceSession(r"model\pangu_weather_24.onnx",
                                     sess_options=options,providers=providers)
for step in range(1,365):
    output_upper,output_surface=ort_session_24.run(None,{'input':input_upper,'input_surface':input_surface})
    input_upper,input_surface=output_upper,output_surface
# Run the inference session
    np.save(os.path.join(output_data_dir, f'z-q-t-u-v_{step}.npy'), output_upper)
    np.save(os.path.join(output_data_dir, f'mslp-10mU-10mV-2mT_{step}.npy'), output_surface)
    epoch_end_time=time.time()
    execution_time=epoch_end_time-epoch_start_time
    print(f"Excution time:{execution_time:5f}seconds")
    start_time=time.time()
    print(f'2020-09-1{step+2}T21 run successfully')
               

 