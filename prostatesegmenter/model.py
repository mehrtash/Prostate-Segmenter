from __future__ import print_function

class CNNModel(object):
    def __init__(self, data_streamer, model):
        self.ds = data_streamer
        self.mean_train = self.ds.load_mean_train()
        self.model = model

    @staticmethod
    def preprocess(input_array, mean_train):
        input_array = input_array.astype('float32')
        input_array -= mean_train
        input_array /= 255.
        return input_array

    def predict_test(self, input_test_slices_arr):
        print("Data preprocessing.")
        test_slices_arr = self.preprocess(input_test_slices_arr, self.mean_train)
        print("Predicting the labelmap.")
        test_slices_prediction = self.model.predict(test_slices_arr, verbose=1)
        return test_slices_prediction
