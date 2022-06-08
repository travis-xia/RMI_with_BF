import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

btree= [[2.4508798122406006,2.4973158836364746,2.4117677211761475,2.3663535118103027,2.2249765396118164],
 [ 5.4627790451049805,4.8769495487213135 , 4.06696343421936,3.311267614364624 , 3.0908925533294678],
[ 5.725619792938232,4.9922943115234375 ,4.53657341003418 ,4.162031888961792 , 3.8200430870056152],
[ 6.787414073944092, 5.9563047885894775,  5.27999472618103,5.253568887710571 , 4.851562023162842],
[ 6.835610866546631, 6.40263557434082,5.175624370574951 , 4.888738632202148, 4.727391481399536]]
new_model=[[4.6331000328063965 ,3.7371103763580322 ,2.5895140171051025,1.4484210014343262,0.45811009407043457],
[5.970656394958496,  4.341186285018921 ,  3.451045274734497 ,2.2382395267486572 ,  0.6615221500396729 ],
[ 5.251078844070435 ,  4.141468048095703,  3.214567184448242, 1.9120266437530518 ,  0.8567922115325928],
[ 5.922184705734253 , 4.617767333984375 , 3.481236219406128 , 2.744631052017212 , 1.304849624633789 ],
[  6.1372339725494385, 4.89045786857605 , 3.714799404144287,   2.4765849113464355, 1.3465287685394287]]
correct_rate_1M = [1,.99983,.99974,.99966,.99955]
correct_rate_25M = [1,.99994,.99991,.99983,.99977]
correct_rate_75M = [1,.99999,.99999,.99999,.99993]
correct_rate_150M = [1,1,1,.99999,1]
correct_rate_200M = [1,1,1,1,1]


#btree和new_model的 准确率-负键占比图
plt.clf()
plt.figure(num=1, figsize=(8, 5))#,facecolor='#eeeeff')
x_label = ['positive','25%','50%','75%','100%']
plt.margins(0) # 控制两边是否有边界
plt.plot(x_label,correct_rate_1M,label="correct_rate_1M")
for a, b in zip(x_label, correct_rate_1M):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,correct_rate_25M,label="correct_rate_25M")
for a, b in zip(x_label, correct_rate_25M):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,correct_rate_75M,label="correct_rate_75M")
for a, b in zip(x_label, correct_rate_75M):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,correct_rate_150M,label="correct_rate_150M")
for a, b in zip(x_label, correct_rate_150M):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,correct_rate_200M,label="correct_rate_200M")
for a, b in zip(x_label, correct_rate_200M):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.xlabel("pencentage of negative keys(%)")
plt.ylabel("accuracy(%)")
#plt.ylim(0,100)
plt.legend()
#plt.show()
plt.savefig('accuracy_negative.jpg', format='jpg', dpi=1000)

#btree和new_model的 效率-数据量图(25%negative)
plt.clf()
x_label = ['1M','25M','75M','150M','200M']
y1 =np.array([])
for i in range(len(btree)):
    y1 = np.append(y1,np.around(btree[i][1]*10,3))
y2 =np.array([])
for i in range(len(new_model)):
    y2 = np.append(y2,np.around(new_model[i][1]*10,3))
plt.margins(0) # 控制两边是否有边界
plt.plot(x_label,y1,label="btree")
for a, b in zip(x_label, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,y2,label="new_model")
for a, b in zip(x_label, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.xlabel("size of dataset(M)")
plt.ylabel("average time_cost of one search(μs)")
plt.ylim(0,100)
plt.legend()
#plt.show()
plt.savefig('speed_size.jpg', format='jpg', dpi=1000)

#btree和new_model的 效率-负键占比图
plt.clf()
x_count = [1,2,3,4,5]
x_label = ['1M','25M','75M','150M','200M']
x_label = ['positive','25%','50%','75%','100%']
y1 = np.around(np.array(btree[4])*10,3)
y2 = np.around(np.array(new_model[4])*10,3)
plt.margins(0) # 控制两边是否有边界
plt.plot(x_label,y1,label="btree")
for a, b in zip(x_label, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.plot(x_label,y2,label="new_model")
for a, b in zip(x_label, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.xlabel("pencentage of negative keys(%)")
plt.ylabel("average time_cost of one search(μs)")
plt.ylim(0,100)
plt.legend()
#plt.show()
plt.savefig('speed_negative.jpg', format='jpg', dpi=1000)

