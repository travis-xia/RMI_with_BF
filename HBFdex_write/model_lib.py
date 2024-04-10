# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures as poly# 导入多项式回归模型
class LinearRegression:
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0

    def fit(self,data,label):
        # data = data.flatten()
        # data_sum = data.sum()
        # label_sum = label.sum()
        # data_mul_label = (data*label).sum()
        # data_square_div_label_sum = (data**2/label_sum).sum()
        # n = len(data)
        # print("  model params:",data_sum,label_sum,data_mul_label,data_square_div_label_sum)
        # self.coef_ = (data_sum / n - data_mul_label / label_sum) / (data_sum / n / label_sum * data_sum - data_square_div_label_sum)
        # self.intercept_ = (label_sum - self.coef_ * data_sum) / n
        temp_model = lr()
        temp_model.fit(data.reshape(-1,1),label)
        self.coef_ = temp_model.coef_
        self.intercept_ = temp_model.intercept_

    def predict(self,base_array):
        base_array = np.array(base_array)
        #print('basearray',base_array)
        return self.coef_ * base_array + self.intercept_

class PolyRegression:
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0
        self.data0 =0

    def fit(self,data,label):
        X = np.hstack([data, (data-data[0])**0.5])
        temp_model = lr()
        temp_model.fit(X,label)
        self.coef_ = temp_model.coef_
        self.intercept_ = temp_model.intercept_
        self.data0 = data[0]
        print(self.coef_,self.intercept_)


    def predict(self,base_array):
        base_array = np.array(base_array)
        # X = np.hstack([base_array, base_array ** 0.5])
        # return np.dot(X,self.coef_)+ self.intercept_
        # if base_array-self.data0<0:
        #     print(base_array,self.data0)
        #     return np.nan
        return base_array*self.coef_[0]+ ((base_array-self.data0)**0.5)*self.coef_[1]+ self.intercept_

if __name__ == "__main__":
    data = np.arange(500)
    label = np.arange(500)*2+5
    model = LinearRegression()
    model.fit(data.reshape(-1,1),label)
    print(model.coef_,model.intercept_)
    print(model.predict([[25]]))
    # print('parameters:', model.coef_, model.intercept_)
    # print(model.predict([ [23] ]))
