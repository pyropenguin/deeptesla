'''
Training Module
'''
from data import load_test_train_data
from model_tf2 import build_model

BATCH_SIZE = 16
EPOCHS = 10

X_train, X_test, Y_train, Y_test = load_test_train_data()

m = build_model(X_train[0].shape)
m.summary()

m.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
m.save('savedmodel')

m.evaluate(X_test, Y_test)