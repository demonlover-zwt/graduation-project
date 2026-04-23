import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def distance(channel_x, channel_y, channel_dep, ship_x, ship_y):
    dis=np.sqrt((channel_x-ship_x)**2+(channel_y-ship_y)**2+channel_dep**2)

    return dis

def angle(front_channel_x, front_channel_y, behind_channel_x, behind_channel_y, channel_dep, ship_x, ship_y):
    channel_x=(front_channel_x+behind_channel_x)/2
    channel_y=(front_channel_y+behind_channel_y)/2
    ship_channel=[channel_x-ship_x, channel_y-ship_y, channel_dep]
    frontchannel_channel=[behind_channel_x-front_channel_x, behind_channel_y-front_channel_y, 0]
    cos_theta=np.dot(ship_channel, frontchannel_channel)/(np.linalg.norm(ship_channel)*np.linalg.norm(frontchannel_channel))

    return cos_theta

def compute_fk_spectrum(data, dt, dx):
    # 二维FFT
    fk_spectrum = np.fft.fft2(data)
    # 移位使得零频在中心
    fk_spectrum_shifted = np.fft.fftshift(fk_spectrum)
    # 幅度归一化
    nt, nx = data.shape
    spectrum = np.abs(fk_spectrum_shifted)/(nt*nx)
    # 生成频率和波数坐标轴
    freq = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx))

    # 筛选正频率（包括零频）
    positive_freq_mask = freq >= 0
    freq_positive = freq[positive_freq_mask]
    spectrum_positive = 2*spectrum[positive_freq_mask, :]
    
    return spectrum_positive, freq_positive, k
    
def sinc(x):

    return np.sin(x)/x

def fk_filter(data, dt, dx, v_min=1400):
    """
    对输入数据应用 FK 滤波，滤掉波速小于 v_min 的信号。

    参数:
    data : 2D numpy array
        输入的空间-时间数据，形状为 (num_samples, num_traces)
    dt : float
        采样时间间隔（秒）
    dx : float
        空间采样间隔（米）
    v_min : float
        最小波速（米/秒）
    
    返回:
    filtered_data : 2D numpy array
        滤波后的数据
    """
    Ns = data.shape[1]
    Nx = data.shape[0]

    f0 = np.fft.fftshift(np.fft.fftfreq(Ns, d=dt))
    k0 = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ft2 = np.fft.fftshift(np.fft.fft2(data))
    F, K = np.meshgrid(f0, k0)

    # 计算相速度，避免除零错误
    phase_velocity = np.abs(F / (K + np.finfo(float).eps))
    # 创建过滤掩码
    filter_mask = (phase_velocity >= v_min)
    # 应用过滤掩码
    fk_data_filtered = ft2 * filter_mask
    # 进行逆 2D FFT 变换回到时空域，并应用fftshift
    filtered_data = np.fft.ifft2(np.fft.ifftshift(fk_data_filtered))
    
    # 返回实部，因为输入数据是实值的
    return np.real(filtered_data)

def distance(channel_x, channel_y, channel_dep, ship_x, ship_y):
    dis=np.sqrt((channel_x-ship_x)**2+(channel_y-ship_y)**2+channel_dep**2)

    return dis

def dis_hor(channel_x, channel_y, ship_x, ship_y):
    dis=np.sqrt((channel_x-ship_x)**2+(channel_y-ship_y)**2)

    return dis

def angle(front_channel_x, front_channel_y, behind_channel_x, behind_channel_y, channel_dep, ship_x, ship_y):
    channel_x=(front_channel_x+behind_channel_x)/2
    channel_y=(front_channel_y+behind_channel_y)/2
    ship_channel=[channel_x-ship_x, channel_y-ship_y, channel_dep]
    frontchannel_channel=[behind_channel_x-front_channel_x, behind_channel_y-front_channel_y, 0]
    cos_theta=np.dot(ship_channel, frontchannel_channel)/(np.linalg.norm(ship_channel)*np.linalg.norm(frontchannel_channel))

    return cos_theta

def angle_v2(front_channel_x, front_channel_y, behind_channel_x, behind_channel_y, channel_dep, ship_x, ship_y):
    channel_x=(front_channel_x+behind_channel_x)/2
    channel_y=(front_channel_y+behind_channel_y)/2
    ship_channel=[channel_x-ship_x, channel_y-ship_y, channel_dep*0]
    frontchannel_channel=[behind_channel_x-front_channel_x, behind_channel_y-front_channel_y, 0]
    cos_theta=np.dot(ship_channel, frontchannel_channel)/(np.linalg.norm(ship_channel)*np.linalg.norm(frontchannel_channel))

    return cos_theta


def sinc(x):

    return np.sin(x)/x

def new_channel(channel_x, channel_y, front_channel_x, front_channel_y, theta, dx, num):
    cos_alpha=(channel_x-front_channel_x)/np.sqrt((channel_x-front_channel_x)**2+(channel_y-front_channel_y)**2)
    sin_alpha=(channel_y-front_channel_y)/np.sqrt((channel_x-front_channel_x)**2+(channel_y-front_channel_y)**2)
    
    cos_theta_alpha=np.cos(theta*np.pi/180)*cos_alpha-np.sin(theta*np.pi/180)*sin_alpha
    sin_theta_alpha=np.cos(theta*np.pi/180)*sin_alpha+np.sin(theta*np.pi/180)*cos_alpha

    ind=np.zeros([2,num])
    for i in range(num):
        ind[0,i]=channel_x+(i+1)*dx*cos_theta_alpha
        ind[1,i]=channel_y+(i+1)*dx*sin_theta_alpha

    return ind

def normolize(fk_spec):
    max_spec=np.max(fk_spec)
    min_spec=np.min(fk_spec)
    nor_spec=(fk_spec-min_spec)/(max_spec-min_spec)

    return nor_spec

### ================== ###

### DAS基本参数
dx=4   ### 道间距，米
fs=500   ### 采样率，赫兹
gauge_length=10   ### 标距，米
dt = 1/fs   # 时间采样间隔（秒）
sound_speed=1500   ### 水中声速，米/秒
depth=22   ### 暂时统一设置通道深度均为22米
deposit_speed=1570   ### P波在沉积层中速度，米/秒
cri_dis=depth*sound_speed/(np.sqrt(deposit_speed**2-sound_speed**2))
cri_dis_water=depth*deposit_speed/(np.sqrt(deposit_speed**2-sound_speed**2))

ch_ini=2192   ### 海缆段的初始通道号
ch_fin=2491   ### 海缆段的终止通道号
num=5  ### 每次定位的通道数

max_fre=49.335
min_fre=48.985
fre_up=np.round(((max_fre+min_fre)/2+(max_fre-min_fre)/2*1.5),3)
fre_down=np.round(((max_fre+min_fre)/2-(max_fre-min_fre)/2*1.5),3)
k_up=np.round((((max_fre+min_fre)/2)/sound_speed*1.5),3)
k_down=np.round((-((max_fre+min_fre)/2)/sound_speed*1.5),3)

### 控制定位段
know_ini=2192
know_fin=2341
pos_ini=2342
pos_fin=pos_ini+num-1

### ======= ###
# 计算实测fk谱
ship_name="CORAL ACROPORA"
signal_all=np.load("./0_辅助数据/"+ship_name+".npy")
signal_all=fk_filter(signal_all, dt, dx)
channel_frag=str(pos_ini)+"-"+str(pos_fin)
os.makedirs("./"+ship_name+"/"+channel_frag, exist_ok=True)

signal_set=np.zeros([np.size(signal_all,0), np.size(signal_all,1)])

signal_set[know_ini-ch_ini:pos_fin-ch_ini+1,:]=signal_all[know_ini-ch_ini:pos_fin-ch_ini+1,:]
# 计算频率-波数谱
spectrum_positive, freq_positive, k = compute_fk_spectrum(np.transpose(signal_set), dt, dx)
dB_fk=np.flipud(20 * np.log10(spectrum_positive+1e-12))

f_start=int(fre_down*len(freq_positive)/(fs/2)) 
f_end=int(fre_up*len(freq_positive)/(fs/2))  
k_start=int((k_down-(-1/(2*dx)))*len(k)*dx) 
k_end=int((k_up-(-1/(2*dx)))*len(k)*dx) 
fk_plot=dB_fk[len(freq_positive)-f_end:len(freq_positive)-f_start,k_start:k_end]

plot_max=10
plot_min=-10
# fk_plot[fk_plot>plot_max]=plot_max
fk_plot[fk_plot<plot_min]=plot_min
mea_fk=normolize(fk_plot)

fig, ax = plt.subplots(figsize=(8, 9))   # 宽度 8 英寸，高度 9 英寸
cax=ax.imshow(fk_plot,aspect='auto', extent=[k[k_start], k[k_end], freq_positive[f_start], freq_positive[f_end]])
cbar = fig.colorbar(cax, ax=ax, fraction=0.025, pad=0.04)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('dB [20log$_1$$_0$((nε/s)·s·m)]', fontsize=16)
plt.xlim(k[k_start], k[k_end])
plt.ylim(freq_positive[f_start], freq_positive[f_end])
plt.xlabel('Wavenumber (m$^-$$^1$)', fontsize=16)
plt.ylabel('Frequency (Hz)', fontsize=16)
ax.tick_params(labelsize=14)

fre_line = [min_fre, max_fre]
for fre in fre_line:
    ax.axhline(y=fre, color='white', linestyle='--', linewidth=2)
ax.axhline(y=49.245, color='#EE7621', linestyle='--', linewidth=1.5)
ax.axhline(y=49.029, color='#9932CC', linestyle='--', linewidth=1.5)
ax.axhline(y=49.011, color='#9932CC', linestyle='--', linewidth=1.5)

ax.axline((0, 0), slope=sound_speed, color='red', linestyle='-', linewidth=2)
ax.axline((0, 0), slope=-sound_speed, color='red', linestyle='-', linewidth=2)
ax.axline((0, 0), slope=15000, color='#FFD700', linestyle='--', linewidth=1.5)
ax.axline((0, 0), slope=-15000, color='#FFD700', linestyle='--', linewidth=1.5)

plt.savefig("./"+ship_name+"/"+channel_frag+"/mea_fk.png", dpi=600, bbox_inches='tight')
plt.close()

correlation = scipy.signal.correlate2d(mea_fk, mea_fk, mode='same')
y0, x0 = np.unravel_index(np.argmax(correlation), correlation.shape)

### ======= ###
# 模拟fk谱
### 生成标准时间
time_len=6   ### 时间窗长度，分钟
time = np.arange(0, 60*time_len+1/fs, 1/fs) 

### 海缆和船舶坐标读取
cable=np.load("./0_辅助数据/海缆坐标.npy")   # 纬度，经度，埋深
cable_y=cable[:,0]
cable_x=cable[:,1]

ship_name="CORAL ACROPORA"
row_ship_frequency=49.16   ### 根据fk估计的原始船舶窄带噪声频率，Hz
correct_ship = np.load("./0_辅助数据/船舶信息_全校正_多径_"+ship_name+".npy", allow_pickle=True)   ### 未插值船舶信息（经度，纬度，时间）

interp_ship=np.full((len(time),np.size(correct_ship,1)-1),np.nan)
for i in range(np.size(correct_ship,1)-1):
    interp_func = interp1d(correct_ship[:,2], correct_ship[:,i], kind='linear', fill_value="extrapolate")
    interp_ship[:,i]=interp_func(time)
ship_x=interp_ship[:,0]
ship_y=interp_ship[:,1]

signal_sim=np.zeros([np.size(signal_all,0), np.size(signal_all,1)+1])   ### 因为模拟是多算了一个6分整的数据，所以需+1，影响不大
### 已知段
ch_know=list(range(know_ini, know_fin+1))
for channel in ch_know:
    signal_sim[channel-ch_ini,:]=np.real(np.load("./"+ship_name+"/know/"+str(channel)+'_simulate_signal.npy'))

pos_ang=list(range(-5,6))
mismatch=np.zeros(len(pos_ang))
mismatch_f=np.zeros(len(pos_ang))
mismatch_k=np.zeros(len(pos_ang))
kk=0
for theta in pos_ang:
    os.makedirs("./"+ship_name+"/"+channel_frag+"/"+str(theta), exist_ok=True)
    channel_ind=new_channel(cable_x[(pos_ini-1)-1], cable_y[(pos_ini-1)-1], cable_x[(pos_ini-1)-2], cable_y[(pos_ini-1)-2], theta, dx, num)

    for j in range(num):
        channel_x=channel_ind[0,j]
        channel_y=channel_ind[1,j]
        if j==0:
            front_channel_x=cable_x[(pos_ini-2)-1]
            front_channel_y=cable_y[(pos_ini-2)-1]
        elif j==1:
            front_channel_x=cable_x[(pos_ini-1)-1]
            front_channel_y=cable_y[(pos_ini-1)-1]
        else:
            front_channel_x=channel_ind[0,j-2]
            front_channel_y=channel_ind[1,j-2]
        
        if j==num-1:
            behind_channel_x=channel_ind[0,j]
            behind_channel_y=channel_ind[1,j]
        elif j==num-2:
            behind_channel_x=channel_ind[0,j+1]
            behind_channel_y=channel_ind[1,j+1]
        else:
            behind_channel_x=channel_ind[0,j+2]
            behind_channel_y=channel_ind[1,j+2]

        ### 计算距离和掠射角
        dis_ch_ship_w=[]
        dis_ch_ship_d=[]
        cos_theta=[]
        for i in range(len(time)):
            hor_dis_ch_ship=dis_hor(channel_x, channel_y, ship_x[i], ship_y[i])
            if hor_dis_ch_ship <= cri_dis:
                dis_ch_ship_w.append(distance(channel_x, channel_y, depth, ship_x[i], ship_y[i]))
                dis_ch_ship_d.append(0)
                cos_theta.append(angle(front_channel_x, front_channel_y, behind_channel_x, behind_channel_y,  depth, ship_x[i], ship_y[i]))

            else:
                dis_ch_ship_w.append(cri_dis_water)
                dis_ch_ship_d.append(hor_dis_ch_ship-cri_dis)
                cos_theta.append(angle_v2(front_channel_x, front_channel_y, behind_channel_x, behind_channel_y,  depth, ship_x[i], ship_y[i]))
            
        dis_ch_ship_w = np.array(dis_ch_ship_w)
        dis_ch_ship_d = np.array(dis_ch_ship_d)
        cos_theta = np.array(cos_theta)

        ### 模拟信号
        ang_fre=2*np.pi*row_ship_frequency
        coe1=1/np.sqrt(dis_ch_ship_w)*(1/(10**((0.5*row_ship_frequency/1000)*dis_ch_ship_d/20)))
        coe2=cos_theta**2
        coe3=sinc(ang_fre*cos_theta*gauge_length/(2*sound_speed))
        signal=coe1*np.exp(ang_fre*1j*(time-dis_ch_ship_w/sound_speed-dis_ch_ship_d/deposit_speed))*coe2*coe3

        signal_sim[pos_ini-ch_ini+j,:]=np.real(signal)

    ### 画模拟fk
    # 计算频率-波数谱
    spectrum_positive, freq_positive, k = compute_fk_spectrum(np.transpose(signal_sim), dt, dx)
    dB_fk=np.flipud(20 * np.log10(spectrum_positive))

    # 绘制频率波数谱
    f_start=int(fre_down*len(freq_positive)/(fs/2)) 
    f_end=int(fre_up*len(freq_positive)/(fs/2))  
    k_start=int((k_down-(-1/(2*dx)))*len(k)*dx) 
    k_end=int((k_up-(-1/(2*dx)))*len(k)*dx) 
    fk_plot=dB_fk[len(freq_positive)-f_end:len(freq_positive)-f_start,k_start:k_end]
    plot_max=-65
    plot_min=-85
    # fk_plot[fk_plot>plot_max]=plot_max
    fk_plot[fk_plot<plot_min]=plot_min
    sim_fk=normolize(fk_plot)

    ### 画fk谱
    fig, ax = plt.subplots(figsize=(8, 9))   # 宽度 8 英寸，高度 9 英寸
    cax=ax.imshow(fk_plot,aspect='auto', extent=[k[k_start], k[k_end], freq_positive[f_start], freq_positive[f_end]])
    cax.set_clim(plot_min, plot_max)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.025, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('dB [20log$_1$$_0$((nε/s)·s·m)]', fontsize=16)
    plt.xlim(k[k_start], k[k_end])
    plt.ylim(freq_positive[f_start], freq_positive[f_end])
    plt.xlabel('Wavenumber (m$^-$$^1$)', fontsize=16)
    plt.ylabel('Frequency (Hz)', fontsize=16)
    ax.tick_params(labelsize=14)

    fre_line = [min_fre, max_fre]
    for fre in fre_line:
        ax.axhline(y=fre, color='white', linestyle='--', linewidth=2)
    ax.axhline(y=49.245, color='#EE7621', linestyle='--', linewidth=1.5)
    ax.axhline(y=49.029, color='#9932CC', linestyle='--', linewidth=1.5)
    ax.axhline(y=49.011, color='#9932CC', linestyle='--', linewidth=1.5)
    ax.axhline(y=49.270, color='#FFC1C1', linestyle='-.', linewidth=1.5)

    ax.axline((0, 0), slope=15000, color='#FFD700', linestyle='--', linewidth=1.5)
    ax.axline((0, 0), slope=-15000, color='#FFD700', linestyle='--', linewidth=1.5)
    ax.axline((0, 0), slope=sound_speed, color='red', linestyle='-', linewidth=2)
    ax.axline((0, 0), slope=-sound_speed, color='red', linestyle='-', linewidth=2)

    plt.savefig("./"+ship_name+"/"+channel_frag+"/"+str(theta)+"/sim_fk.png", dpi=600, bbox_inches='tight')
    plt.close()

    correlation = scipy.signal.correlate2d(sim_fk, mea_fk, mode='same')
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    mismatch[kk]=np.round(np.sqrt((y-y0)**2+(x-x0)**2), 3)
    mismatch_f[kk]=y-y0
    mismatch_k[kk]=x-x0
    kk=kk+1

fig = plt.figure(figsize=[12, 9])
plt.plot(pos_ang, np.abs(mismatch), linestyle='-', color='r', linewidth=2, label='combine deviation', alpha=0.5)
plt.plot(pos_ang, np.abs(mismatch_f), linestyle='--', color='b', linewidth=2, label='frequency deviation', alpha=0.5)
plt.plot(pos_ang, np.abs(mismatch_k), linestyle='-.', color='g', linewidth=1, label='wavenumber deviation', alpha=1)
plt.xlim(pos_ang[0], pos_ang[-1])
plt.ylim(0, 1.1*np.nanmax(mismatch))
plt.xlabel("Angle [°]", fontsize=16)
plt.ylabel("Deviation", fontsize=16)
plt.tick_params(labelsize=14)
plt.legend()
plt.savefig("./"+ship_name+"/"+channel_frag+"/deviation.png", dpi=600, bbox_inches='tight')
plt.close()
