import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
btree= [[2.6737821102142334,2.678514003753662,2.5793087482452393,2.515442132949829,2.430807590484619],
 [ 5.4627790451049805,4.8769495487213135 , 4.06696343421936,3.311267614364624 , 3.0908925533294678],
[ 5.725619792938232,4.9922943115234375 ,4.53657341003418 ,4.162031888961792 , 3.8200430870056152],
[ 6.787414073944092, 5.9563047885894775,  5.27999472618103,5.253568887710571 , 4.851562023162842],
[ 6.835610866546631, 6.40263557434082,5.175624370574951 , 4.888738632202148, 4.727391481399536]]
new_model=[[2.7088754177093506 ,2.119218349456787 ,1.541771650314331,0.9559693336486816,0.3732757568359375],
[3.061488389968872,  2.4472806453704834 ,  1.8192663192749023 ,1.2072980403900146 ,  0.5634274482727051 ],
[ 3.2982606887817383 ,  2.6827011108398438,  2.1104769706726074, 1.5119593143463135 , 0.9069156646728516],
[ 3.319856643676758 , 2.7598958015441895, 2.2679896354675293  , 1.762986660003662 , 1.2561805248260498 ],
[  3.366424322128296, 2.8162856101989746 , 2.314316987991333,  1.8419668674468994,  1.362917184829712]]
RMI = [
[3.165493727	,3.168004036	,3.218780279	,3.223211765	,3.248328447],
[3.437907696	,3.426084042	,3.498665094	,3.536888123	,3.568657398],
[3.909689188	,3.924142599	,3.980231762	,4.007567644	,4.055919647],
[3.959991932	,3.934695721	,3.99259758	,4.040940523	,4.091480017],
[4.565482616	,4.592906952	,4.657893658	,4.719358444	,4.778594971]
]
opt = [
 [2.5958282947540283, 2.118424415588379 ,1.610316514968872  ,1.1272315979003906,  0.6326954364776611],
 [ 3.002242088317871 , 2.478701114654541 , 1.9241831302642822  , 1.3667888641357422 ,  0.807429313659668],
[ 3.1354079246520996 , 2.625656843185425 , 2.1037650108337402  , 1.5858478546142578 , 1.040748119354248 ],
[ 3.1354079246520996 , 2.625656843185425 , 2.1037650108337402  , 1.5858478546142578 , 1.040748119354248 ],
[ 3.1354079246520996 , 2.625656843185425 , 2.1037650108337402  , 1.5858478546142578 , 1.040748119354248 ]]
alex = [
[26.38009638	,31.36600838	,38.03991938	,41.52492366],
[25.84373964	,32.8468109	,36.69085312	,37.0598187],
[25.55509973	,34.61518654	,35.75340652	,43.47026224]
]
rbt = [[2.930983067,3.19965291,3.575310707],
 [2.902566671,3.149635792,3.476439714],
 [2.323414373,2.739169121,4.683186531],
 [2.834595919,2.816935539,2.648660898]]

correct_rate_1M = [1,.99983,.99974,.99966,.99955]
correct_rate_25M = [1,.99994,.99991,.99983,.99977]
correct_rate_75M = [1,.99999,.99999,.99999,.99993]
correct_rate_150M = [1,1,1,.99999,1]
correct_rate_200M = [1,1,1,1,1]
mem_array =[[64753.4062,53751.3047	,34976.5312,	13797.2383],
[50536.7578,	37914.1211,	18974.4453,	6347.4414],
[885.0195,	691.2305,	401.3789	,212.9336],
[773.3844,	577.2263,	286.9319	,95.9058],
[997.0625,	747.9726,	374.5195	,125.4805]]

# print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast',
#  'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind',
#  'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
#  'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks',
#  'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']


#mit数据集 0%,50%,100% 效率-数据集大小图
title_string = ['0%','50%','100%']
for i in [0,2,4]:
 plt.clf()
 plt.style.use('tableau-colorblind10')
 y_b = [10*btree[x][i] for x in range(1,5)]
 y_rmi = [10*RMI[x][i] for x in range(1, 5)]
 y_alex_ = [alex[i//2][x] for x in range(4)]
 y_hbf = [10*opt[x][i] for x in range(1, 5)]
 # 构建数据
 x = np.arange(4)
 bar_width  = 0.15
 tick_label = ['25M', '75M', '150M', '200M']
 plt.bar(x, y_b, bar_width, align="center", tick_label=tick_label, label='Btree')
 plt.bar(x + 1 * bar_width, y_rmi, bar_width, align="center", tick_label=tick_label, label='RMI')
 plt.bar(x + 2 * bar_width, y_alex_, bar_width, align="center", tick_label=tick_label, label='ALEX',color = 'skyblue')
 plt.bar(x + 3 * bar_width, y_hbf, bar_width, align="center", tick_label=tick_label, label='HBFdex',color = 'grey')
 # 添加标题和标签
 # plt.title('MIT_BOOK_INT({})'.format(title_string[i//2]))
 plt.title('MIT_BOOK_INT')
 plt.xlabel('Data size')
 plt.ylabel('average time_cost of one search(μs)')
 # 设置x轴刻度显示位置
 plt.xticks(x+bar_width/2, tick_label)
 plt.legend(loc='upper left')
 # 使用对数尺度
 plt.show()

# #response中的rbtree
# b_response = [68.35610867,51.75624371,47.27391481]
# rbt_response = [29.30983067,31.9965291,35.75310707]
# rmi_response = [45.65482616,46.57893658,47.78594971]
# alex_response  = [41.52492366,37.0598187,43.47026224]
# hbf_response = [25.40533066,18.39705229	,11.84376001]
# labels = ['Btree', 'RBtree', 'RMI', 'ALEX', 'HBFdex']
# plt.style.use('tableau-colorblind10')
# # 构建数据
# x = np.arange(3)
# bar_width  = 0.15
# tick_label = ['0%', '50%', '100%']
# plt.bar(x, b_response, bar_width, align="center", tick_label=tick_label, label='Btree')
# plt.bar(x + bar_width, rbt_response, bar_width, align="center", tick_label=tick_label, label='RBtree')
# plt.bar(x + 2 * bar_width, rmi_response, bar_width, align="center", tick_label=tick_label, label='RMI')
# plt.bar(x + 3 * bar_width, alex_response, bar_width, align="center", tick_label=tick_label, label='ALEX')
# plt.bar(x + 4 * bar_width, hbf_response, bar_width, align="center", tick_label=tick_label, label='HBFdex')
# # 添加标题和标签
# plt.title('MIT_BOOK_INT')
# plt.xlabel('negtive rate')
# plt.ylabel('average time_cost of one search(μs)')
# # 设置x轴刻度显示位置
# plt.xticks(x+bar_width/2, tick_label)
# plt.legend(loc='upper right')
# plt.show()

# #'Btree', 'RMI', 'ALEX', 'HBFdex'内存-数据及大小图
# plt.clf()
# labels = ['Btree',  'RMI', 'ALEX', 'HBFdex']
# plt.style.use('tableau-colorblind10')
# # 构建数据
# x = np.arange(4)
# y1 = mem_array[0]
# y3 =mem_array[2]
# y4 =mem_array[3]
# y5 =mem_array[4]
# y1.reverse()
# y3.reverse()
# y4.reverse()
# y5.reverse()
# # 将内存数据转换为对数尺度
# log_y1 = [round(10**(-2)*i, 2) for i in y1]
# log_y3 = [round(10**(-2)*i, 2) for i in y3]
# log_y4 = [round(10**(-2)*i, 2) for i in y4]
# log_y5 = [round(10**(-2)*i, 2) for i in y5]
#
# bar_width  = 0.2
# tick_label = ['25M', '75M', '150M', '200M']
# plt.bar(x, log_y1, bar_width, align="center", tick_label=tick_label, label='Btree')
# plt.bar(x + 1 * bar_width, log_y3, bar_width, align="center", tick_label=tick_label, label='RMI')
# plt.bar(x + 2 * bar_width, log_y4, bar_width, align="center", tick_label=tick_label, label='ALEX')
# plt.bar(x + 3 * bar_width, log_y5, bar_width, align="center", tick_label=tick_label, label='HBFdex')
# # 添加标题和标签
# plt.title('Memory Usage')
# plt.xlabel('Data size')
# plt.ylabel('Memory/100MB')
# # 设置x轴刻度显示位置
# plt.xticks(x+bar_width/2, tick_label)
# plt.legend(loc='upper left')
# # 使用对数尺度
# plt.yscale('log')
# plt.show()


# #'Btree', 'RBtree', 'RMI', 'ALEX', 'HBFdex'内存-数据及大小图
# plt.clf()
# labels = ['Btree', 'RBtree', 'RMI', 'ALEX', 'HBFdex']
# plt.style.use('tableau-colorblind10')
# # 构建数据
# x = np.arange(4)
# y1 = mem_array[0]
# y2 = mem_array[1]
# y3 =mem_array[2]
# y4 =mem_array[3]
# y5 =mem_array[4]
# y1.reverse()
# y2.reverse()
# y3.reverse()
# y4.reverse()
# y5.reverse()
# # 将内存数据转换为对数尺度
# log_y1 = [round(10**(-2)*i, 2) for i in y1]
# log_y2 = [round(10**(-2)*i, 2) for i in y2]
# log_y3 = [round(10**(-2)*i, 2) for i in y3]
# log_y4 = [round(10**(-2)*i, 2) for i in y4]
# log_y5 = [round(10**(-2)*i, 2) for i in y5]
#
# print(log_y4)
# bar_width  = 0.15
# tick_label = ['25M', '75M', '150M', '200M']
# plt.bar(x, log_y1, bar_width, align="center", tick_label=tick_label, label='Btree')
# plt.bar(x + bar_width, log_y2, bar_width, align="center", tick_label=tick_label, label='RBtree')
# plt.bar(x + 2 * bar_width, log_y3, bar_width, align="center", tick_label=tick_label, label='RMI')
# plt.bar(x + 3 * bar_width, log_y4, bar_width, align="center", tick_label=tick_label, label='ALEX')
# plt.bar(x + 4 * bar_width, log_y5, bar_width, align="center", tick_label=tick_label, label='HBFdex')
# # 添加标题和标签
# plt.title('Memory Usage')
# plt.xlabel('Data size')
# plt.ylabel('Memory/100MB')
# # 设置x轴刻度显示位置
# plt.xticks(x+bar_width/2, tick_label)
# plt.legend(loc='upper left')
# # 使用对数尺度
# plt.yscale('log')
# plt.show()


# #btree和new_model的 准确率-负键占比图
# plt.clf()
# plt.figure(num=1, figsize=(8, 5))#,facecolor='#eeeeff')
# x_label = ['positive','25%','50%','75%','100%']
# plt.margins(0) # 控制两边是否有边界
# plt.plot(x_label,correct_rate_1M,label="1M")
# for a, b in zip(x_label, correct_rate_1M):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,correct_rate_25M,label="25M")
# for a, b in zip(x_label, correct_rate_25M):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,correct_rate_75M,label="75M")
# for a, b in zip(x_label, correct_rate_75M):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,correct_rate_150M,label="150M")
# for a, b in zip(x_label, correct_rate_150M):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,correct_rate_200M,label="200M")
# for a, b in zip(x_label, correct_rate_200M):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.xlabel("pencentage of negative keys(%)")
# plt.ylabel("accuracy(%)")
# plt.ylim(0.999,1.0002)
# plt.legend(loc='lower right')
# #plt.show()
# plt.savefig('accuracy_negative.jpg', format='jpg', dpi=1000)

# # btree和new_model的 效率-数据量图(25%negative)
# plt.clf()
# x_label = [1,25,75,150,200]
# y1 =np.array([])
# for i in range(len(btree)):
#     y1 = np.append(y1,np.around(btree[i][1]*10,3))
# y2 =np.array([])
# for i in range(len(new_model)):
#     y2 = np.append(y2,np.around(new_model[i][1]*10,3))
# y3 =np.array([])
# for i in range(len(RMI)):
#     y3 = np.append(y3,np.around(RMI[i][1]*10,3))
# y4 =np.array([])
# for i in range(len(RMI)):
#     y4 = np.append(y4,np.around(opt[i][1]*10,3))
# plt.margins(0) # 控制两边是否有边界
# plt.plot(x_label,y1,label="btree")
# # for a, b in zip(x_label, y1):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,y2,label="new_model")
# # for a, b in zip(x_label, y2):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,y3,label="RMI")
# # for a, b in zip(x_label, y3):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,y3,label="optimized")
# # for a, b in zip(x_label, y4):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.xlabel("size of dataset(M)")
# plt.ylabel("average time_cost of one search(μs)")
# plt.ylim(0,100)
# plt.legend()
# plt.show()
# # plt.savefig('speed_size.jpg', format='jpg', dpi=1000)

# #btree和new_model的 效率-负键占比图
# plt.clf()
# x_count = [1,2,3,4,5]
# x_label = ['1M','25M','75M','150M','200M']
# x_label = ['positive','25%','50%','75%','100%']
# y1 = np.around(np.array(btree[4])*10,3)
# y2 = np.around(np.array(new_model[4])*10,3)
# plt.margins(0) # 控制两边是否有边界
# plt.plot(x_label,y1,label="btree")
# for a, b in zip(x_label, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.plot(x_label,y2,label="new_model")
# for a, b in zip(x_label, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.xlabel("pencentage of negative keys(%)")
# plt.ylabel("average time_cost of one search(μs)")
# plt.ylim(0,100)
# plt.legend()
# #plt.show()
# plt.savefig('speed_negative.jpg', format='jpg', dpi=1000)

