import numpy as np

class LinearRegression:
    def __init__(self):
        self.weight = 0
        self.bias = 0

    def fit(self,data,label):
        data = data.flatten()
        data_sum = data.sum()
        label_sum = label.sum()
        data_mul_label = (data*label).sum()
        data_square_div_label_sum = (data**2/label_sum).sum()
        n = len(data)
        print(data_sum,label_sum,data_mul_label,data_square_div_label_sum)
        self.weight = (data_sum  / n - data_mul_label/label_sum) / (data_sum / n/label_sum *data_sum- data_square_div_label_sum)
        self.bias = (label_sum - self.weight * data_sum) / n


    def predict(self,base_array):
        base_array = np.array(base_array)
        #print('basearray',base_array)
        return self.weight*base_array+self.bias



if __name__ == "__main__":
    data = np.arange(500)
    label = np.arange(500)
    model = LinearRegression()
    model.fit(data,label)
    print('parameters:',model.weight,model.bias)
    print(model.predict([ [23] ]))
