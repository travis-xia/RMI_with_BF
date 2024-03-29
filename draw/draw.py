import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('classic')
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
RMI=[[2.904402494430542 ,2.875233745574951 ,2.9285364151000977,2.9175074100494385,2.9216926097869873],
[3.028968095779419,  3.033641290664673, 3.014747142791748 ,3.0309813022613525 ,  3.038686990737915],
[ 3.2982606887817383 ,  3.2982606887817383,  2.1104769706726074, 1.5119593143463135 , 0.9069156646728516],
[ 3.319856643676758 , 3.319856643676758, 2.2679896354675293  , 1.762986660003662 , 1.2561805248260498 ],
[  3.366424322128296, 3.366424322128296 , 2.314316987991333,  1.8419668674468994,  1.362917184829712]]
opt = [
 [2.5958282947540283, 2.118424415588379 ,1.610316514968872  ,1.1272315979003906,  0.6326954364776611],
 [ 3.002242088317871 , 2.478701114654541 , 1.9241831302642822  , 1.3667888641357422 ,  0.807429313659668],
[ 3.1354079246520996 , 2.625656843185425 , 2.1037650108337402  , 1.5858478546142578 , 1.040748119354248 ],
[ 3.1354079246520996 , 2.625656843185425 , 2.1037650108337402  , 1.5858478546142578 , 1.040748119354248 ],
[ 3.1354079246520996 , 2.625656843185425 , 2.1037650108337402  , 1.5858478546142578 , 1.040748119354248 ]]
correct_rate_1M = [1,.99983,.99974,.99966,.99955]
correct_rate_25M = [1,.99994,.99991,.99983,.99977]
correct_rate_75M = [1,.99999,.99999,.99999,.99993]
correct_rate_150M = [1,1,1,.99999,1]
correct_rate_200M = [1,1,1,1,1]

# print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast',
#  'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind',
#  'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
#  'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks',
#  'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']


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

# btree和new_model的 效率-数据量图(25%negative)
plt.clf()
x_label = [1,25,75,150,200]
y1 =np.array([])
for i in range(len(btree)):
    y1 = np.append(y1,np.around(btree[i][1]*10,3))
y2 =np.array([])
for i in range(len(new_model)):
    y2 = np.append(y2,np.around(new_model[i][1]*10,3))
y3 =np.array([])
for i in range(len(RMI)):
    y3 = np.append(y3,np.around(RMI[i][1]*10,3))
y4 =np.array([])
for i in range(len(RMI)):
    y4 = np.append(y4,np.around(opt[i][1]*10,3))
plt.margins(0) # 控制两边是否有边界
plt.plot(x_label,y1,label="btree")
# for a, b in zip(x_label, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,y2,label="new_model")
# for a, b in zip(x_label, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,y3,label="RMI")
# for a, b in zip(x_label, y3):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,y3,label="optimized")
# for a, b in zip(x_label, y4):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.xlabel("size of dataset(M)")
plt.ylabel("average time_cost of one search(μs)")
plt.ylim(0,100)
plt.legend()
plt.show()
# plt.savefig('speed_size.jpg', format='jpg', dpi=1000)

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

