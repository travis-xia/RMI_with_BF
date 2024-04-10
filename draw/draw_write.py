import numpy as np
from matplotlib import pyplot as plt












hbf_write = [[3.1895952224731445,2.562964677810669,2.0155391693115234,1.3438518047332764, 0.6582329273223877],
[3.868196487426758,2.9612510204315186,2.113172769546509,1.372650384902954,0.6474177837371826],
[4.246711730957031,4.242785692214966,2.4249329566955566, 1.560443639755249,0.6975283622741699],
[4.300432920455933,4.62215518951416,2.256174325942993, 1.503952980041504, 0.657526969909668]]
for i in range(4):
    for j in range(5):
        hbf_write[i][j] = round(hbf_write[i][j]*10,4)
print(hbf_write)
plt.style.use('tableau-colorblind10')
# 构建数据
x = np.arange(5)
y1 = hbf_write[0]
y2 =hbf_write[1]
y3 =hbf_write[2]
y4 =hbf_write[3]
# y1.reverse()
# y2.reverse()
# y3.reverse()
# y4.reverse()

bar_width  = 0.2
tick_label = ['0%','25%', '50%', '75%', '100%']
plt.bar(x, y1, bar_width, align="center", tick_label=tick_label, label='read-only')
plt.bar(x + 1 * bar_width, y2, bar_width, align="center", tick_label=tick_label, label='read-heavy')
plt.bar(x + 2 * bar_width, y3, bar_width, align="center", tick_label=tick_label, label='read-write')
plt.bar(x + 3 * bar_width, y4, bar_width, align="center", tick_label=tick_label, label='write-heavy')
# 添加标题和标签
# plt.title('Memory Usage')
plt.xlabel('negtive rate')
plt.ylabel('average time_cost of one search(μs)')
# 设置x轴刻度显示位置
plt.xticks(x+bar_width/2, tick_label)
plt.legend(loc='upper right')
plt.show()












