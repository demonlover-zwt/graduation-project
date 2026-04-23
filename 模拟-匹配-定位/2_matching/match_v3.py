import os
import numpy as np
import scipy.signal

def normolize(fk_spec):
    max_spec=np.max(fk_spec)
    min_spec=np.min(fk_spec)
    nor_spec=(fk_spec-min_spec)/(max_spec-min_spec)

    return nor_spec

### ===== ###

mea_fk_t=normolize(np.load("./实测fk谱/实测_fk（截取）.npy"))
mea_fk_f=normolize(np.load("./实测fk谱/实测_fk（未截取）.npy"))

ny=np.size(mea_fk_t,0)
nx=np.size(mea_fk_t,1)

correlation = scipy.signal.correlate2d(mea_fk_t, mea_fk_t, mode='same')
y0, x0 = np.unravel_index(np.argmax(correlation), correlation.shape)

ship_name="CORAL ACROPORA"
sim_path="../1_模拟方案/"
# sim_way="1_未校正位置"
# suppose="球面波"
method="二维互相关（归一）"
for sim_way in ["1_不校正位置", "2_全校正位置", "3_全校正位置_多径"]:
    for suppose in ["球面波", "柱面波"]:
        sim_fk_t=normolize(np.load(sim_path+sim_way+"/3d_"+ship_name+"_"+suppose+"_fk（截取）.npy"))
        sim_fk_f=normolize(np.load(sim_path+sim_way+"/3d_"+ship_name+"_"+suppose+"_fk（未截取）.npy"))

        file_path="./"+method+"/"+ship_name+"_"+sim_way+"_"+suppose+".txt"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:

            ###
            correlation = scipy.signal.correlate2d(sim_fk_t, mea_fk_t, mode='same')
            y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
            mismatch=np.round(np.sqrt((y-y0)**2+(x-x0)**2), 3)
            mismatch_f=y-y0
            mismatch_k=x-x0
            text_to_write=f"实测（截取）与模拟（截取）的不匹配度为：{mismatch}；频域不匹配度为：{mismatch_f}；波数域不匹配度为：{mismatch_k}"
            file.write(text_to_write)

            ###
            correlation = scipy.signal.correlate2d(sim_fk_f, mea_fk_t, mode='same')
            y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
            mismatch=np.round(np.sqrt((y-y0)**2+(x-x0)**2), 3)
            mismatch_f=y-y0
            mismatch_k=x-x0
            text_to_write=f"\n实测（截取）与模拟（未截取）的不匹配度为：{mismatch}；频域不匹配度为：{mismatch_f}；波数域不匹配度为：{mismatch_k}"
            file.write(text_to_write)

            ###
            correlation = scipy.signal.correlate2d(sim_fk_f, mea_fk_f, mode='same')
            y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
            mismatch=np.round(np.sqrt((y-y0)**2+(x-x0)**2), 3)
            mismatch_f=y-y0
            mismatch_k=x-x0
            text_to_write=f"\n实测（未截取）与模拟（未截取）的不匹配度为：{mismatch}；频域不匹配度为：{mismatch_f}；波数域不匹配度为：{mismatch_k}"
            file.write(text_to_write)

