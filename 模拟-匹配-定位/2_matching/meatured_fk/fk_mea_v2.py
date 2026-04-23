import numpy as np
import matplotlib.pyplot as plt

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
    
### ================== ###

### DAS基本参数
dx=4   ### 道间距，米
fs=500   ### 采样率，赫兹
gauge_length=10   ### 标距，米
dt = 1/fs   # 时间采样间隔（秒）
sound_speed=1500   ### 水中声速，米/秒
depth=22   ### 暂时统一设置通道深度均为22米

max_fre=49.335
min_fre=48.985
fre_up=np.round(((max_fre+min_fre)/2+(max_fre-min_fre)/2*1.5),3)
fre_down=np.round(((max_fre+min_fre)/2-(max_fre-min_fre)/2*1.5),3)
k_up=np.round((((max_fre+min_fre)/2)/sound_speed*1.5),3)
k_down=np.round((-((max_fre+min_fre)/2)/sound_speed*1.5),3)

### 生成标准时间
time_len=6   ### 时间窗长度，分钟
time = np.arange(0, 60*time_len, 1/fs) 

ship_name="CORAL ACROPORA"
signal_set=np.load("./"+ship_name+".npy")
signal_set=fk_filter(signal_set, dt, dx)
# signal_set=signal_set/np.max(np.abs(signal_set))

# channel_start=2192
# channel_end=2491
# times = [x / 500 for x in range(0, 6*60*fs+1)]
# channels=list(range(channel_start,channel_end+1))
# time_start=0*60*fs
# time_end=6*60*fs
# c_start=0
# c_end=299
# data_plot=np.abs(signal_set)
# thr_u=2700
# thr_d=1300
# data_plot[data_plot>thr_u]=thr_u
# data_plot[data_plot<thr_d]=thr_d

# ### 竖版瀑布图
# fig, ax = plt.subplots(figsize=(9, 6))
# cax=ax.imshow(data_plot,aspect='auto',extent=[times[time_start],times[time_end],channels[c_end],channels[c_start]], cmap='Oranges')
# cax.set_clim(np.min(data_plot), np.max(data_plot))
# cbar = fig.colorbar(cax, ax=ax, fraction=0.015, pad=0.04)
# cbar.ax.tick_params(labelsize=14)
# cbar.set_label('Amplitude', fontsize=16)
# plt.xlim(times[time_start], times[time_end])
# plt.ylim(channels[c_end], channels[c_start]) 
# plt.xlabel('Time (s)', fontsize=16)
# plt.ylabel('Channel', fontsize=16)
# ax.tick_params(labelsize=14)
# plt.savefig("./实测_td.png", dpi=600, bbox_inches='tight')
# plt.close()

signal_set[150:,:]=0
# 计算频率-波数谱
spectrum_positive, freq_positive, k = compute_fk_spectrum(np.transpose(signal_set), dt, dx)
dB_fk=np.flipud(20 * np.log10(spectrum_positive+1e-12))

# 绘制频率波数谱
f_start=int(fre_down*len(freq_positive)/(fs/2)) 
f_end=int(fre_up*len(freq_positive)/(fs/2))  
k_start=int((k_down-(-1/(2*dx)))*len(k)*dx) 
k_end=int((k_up-(-1/(2*dx)))*len(k)*dx) 
fk_plot=dB_fk[len(freq_positive)-f_end:len(freq_positive)-f_start,k_start:k_end]
# fk_plot=dB_fk[len(freq_positive)-f_end-1:len(freq_positive)-f_start,k_start:k_end+1]

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

plt.savefig("./实测_fk（未截取）_v2.png", dpi=600, bbox_inches='tight')
plt.close()
# np.save("实测_fk（未截取）.npy",fk_plot)

plot_max=10
plot_min=-10
# fk_plot[fk_plot>plot_max]=plot_max
fk_plot[fk_plot<plot_min]=plot_min

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

ax.axline((0, 0), slope=sound_speed, color='red', linestyle='-', linewidth=2)
ax.axline((0, 0), slope=-sound_speed, color='red', linestyle='-', linewidth=2)
ax.axline((0, 0), slope=15000, color='#FFD700', linestyle='--', linewidth=1.5)
ax.axline((0, 0), slope=-15000, color='#FFD700', linestyle='--', linewidth=1.5)

plt.savefig("./实测_fk（截取）_v2.png", dpi=600, bbox_inches='tight')
plt.close()
# np.save("实测_fk（截取）.npy",fk_plot)
