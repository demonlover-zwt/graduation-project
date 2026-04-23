import os
import numpy as np

def euc_dis(mea, sim):
    ny=np.size(mea,0)
    nx=np.size(mea,1)
    dis=0
    for i in range(ny):
        for j in range(nx):
            dis=dis+(mea[i,j]-sim[i,j])**2
    
    return np.sqrt(dis)/(ny*nx)

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

ship_name="CORAL ACROPORA"
sim_path="../1_模拟方案/"
# sim_way="1_未校正位置"
# suppose="球面波"
method="欧几里得距离（归一）"
for sim_way in ["1_不校正位置", "2_全校正位置"]:
    for suppose in ["球面波", "柱面波"]:
        sim_fk_t=normolize(np.load(sim_path+sim_way+"/3d_"+ship_name+"_"+suppose+"_fk（截取）.npy"))
        sim_fk_f=normolize(np.load(sim_path+sim_way+"/3d_"+ship_name+"_"+suppose+"_fk（未截取）.npy"))

        file_path="./"+method+"/"+ship_name+"_"+sim_way+"_"+suppose+".txt"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:

            ###
            mismatch=np.round(euc_dis(mea_fk_t, sim_fk_t)*1000, 3)
            text_to_write=f"实测（截取）与模拟（截取）的不匹配度为：{mismatch}‰"
            file.write(text_to_write)

            ###
            mismatch=np.round(euc_dis(mea_fk_t, sim_fk_f)*1000, 3)
            text_to_write=f"\n实测（截取）与模拟（未截取）的不匹配度为：{mismatch}‰"
            file.write(text_to_write)

            ###
            mismatch=np.round(euc_dis(mea_fk_f, sim_fk_f)*1000, 3)
            text_to_write=f"\n实测（未截取）与模拟（未截取）的不匹配度为：{mismatch}‰"
            file.write(text_to_write)

