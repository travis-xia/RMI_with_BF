# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression as lr
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



if __name__ == "__main__":
    data = np.arange(500)
    label = np.arange(500)
    model = LinearRegression()
    model.fit(data,label)
    print('parameters:', model.coef_, model.intercept_)
    print(model.predict([ [23] ]))
