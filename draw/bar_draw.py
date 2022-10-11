import matplotlib.pyplot as plt
import matplotlib
import numpy as np
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
# 设置中文字体
matplotlib.rcParams["font.sans-serif"]=["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 构建数据
x = np.arange(5)
y1 = 100000.0/np.array(btree)[:,1]
y2 = 100000.0/np.array(RMI)[:,1]
y3 = 100000.0/np.array(new_model)[:,1]
bar_width  = 0.25
tick_label = ["1M", '25M', '75M', '150M', '200M']
# 绘制柱状图
# plt.figure(figsize=(4, 4))
plt.style.use('classic')
print(plt.style.available)
plt.bar(x, y1, bar_width, align="center",  tick_label=tick_label, label='btree')
plt.bar(x+bar_width, y2, bar_width, align="center",  tick_label=tick_label, label='RMI')
plt.bar(x+2*bar_width, y3, bar_width, align="center",  tick_label=tick_label, label='new_model')

plt.xlabel("Data size")
plt.ylabel("average time_cost of one search(μs)")
plt.ylabel("average search times in 1s")
# 设置x轴刻度显示位置
plt.xticks(x+bar_width/2, tick_label)

plt.legend(loc='upper right')
plt.show()
#
# for style in plt.style.available:
#  plt.clf()
#  plt.style.use(style)
#  plt.bar(x, y1, bar_width, align="center", tick_label=tick_label, label='btree')
#  plt.bar(x + bar_width, y2, bar_width, align="center", tick_label=tick_label, label='RMI')
#  plt.bar(x + 2 * bar_width, y3, bar_width, align="center", tick_label=tick_label, label='new_model')
#  plt.xlabel("Data size")
#  plt.ylabel("average time_cost of one search(μs)")
#  # 设置x轴刻度显示位置
#  plt.xticks(x + bar_width / 2, tick_label)
#
#  plt.legend(loc='upper left')
#  plt.savefig("./bar_style/"+style+".png")