import pickle

file_name = 'embedding_model.pickle'
loaded_model = pickle.load(open(file_name, "rb"))

X_test_sample = [[1, 2, 20, 4, 1, 5, 0, 0, 1, 3]]
y_predict_test = loaded_model.predict(X_test_sample)

print(y_predict_test)

