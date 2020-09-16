from config import TOP55_SORT, data_len
from data_help import build_data
import numpy as np
from keras.layers import Input, LSTM, Embedding, concatenate, Dense, Dropout, \
    GlobalMaxPool1D, Bidirectional
from keras import optimizers
from keras.models import Model
from wight_emb import load_char_emb
import tensorflow as tf

em_weight = load_char_emb()


def build_model(top):
    """
    build model
    @top: target list
    """
    with tf.device('/gpu:%d' % 0):
        print('building model...')
        char_input = Input(shape=(data_len,), name='char_input')
        char_embedding_layer = Embedding(input_dim=em_weight.shape[0], output_dim=256, input_length=data_len,
                                         weights=[em_weight], trainable=False)

        char_embedding_sequences = char_embedding_layer(char_input)

        maxpool_0 = Bidirectional(LSTM(units=256, return_sequences=True))(char_embedding_sequences)
        maxpool_0 = GlobalMaxPool1D()(maxpool_0)
        maxpool_0 = Dropout(0.5)(maxpool_0)

        model_final = Dense(len(top), activation='softmax')(maxpool_0)
        input_time = Input((10,), name='input_time')
        new_data = concatenate([model_final, input_time], axis=-1)
        model_final2 = Dense(256)(new_data)
        model_final2 = Dropout(0.5)(model_final2)
        model_final = Dense(len(top), activation='softmax')(model_final2)
        model = Model(input=[char_input, input_time], output=model_final)
        adam = optimizers.adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['categorical_accuracy'])

        return model


def build_train_test_data(top):
    """
    build data
    top: target list
    """
    x_all = []
    x_c = []
    for name in top:
        # The number of emails obtained by each person,
        # 10 persons in the paper correspond to 5000. Other cases correspond to 3000.
        data = build_data(name)[0:5000]
        x_all.append(data)
        x_c.append(len(data))
    x_train_l = []
    x_test_l = []
    for x in x_c:
        x_train_l.append(int(x * 0.9) - 1)
        x_test_l.append(x - (int(x * 0.9) - 1))
    i = 0
    x_content2_train = []
    x_time2_train_ = []
    x_content2_test = []
    x_time2_test_ = []
    y_train = np.zeros((sum(x_train_l), len(top)))
    y_test = np.zeros((sum(x_test_l), len(top)))
    test = np.zeros((sum(x_c), len(top)))
    hang_test = 0
    hang_count_train = 0
    hang_count_test = 0
    while i < len(top):
        print(i)
        current_data_train = x_all[i][0:x_train_l[i]]
        current_data_test = x_all[i][x_train_l[i]:]
        for d in current_data_train:
            x_content2_train.append(d[0][0])
            x_time2_train_.append(d[1:11])
        y_train[hang_count_train:hang_count_train + x_train_l[i], i] = 1
        test[hang_test:hang_test + x_train_l[i], i] = 1
        hang_test += x_train_l[i]
        for d in current_data_test:
            x_content2_test.append(d[0][0])
            x_time2_test_.append(d[1:11])
        y_test[hang_count_test:hang_count_test + x_test_l[i], i] = 1
        test[hang_test:hang_test + x_test_l[i], i] = 1
        hang_test += x_test_l[i]
        hang_count_train += x_train_l[i]
        hang_count_test += x_test_l[i]
        i += 1
    x_content_train = np.array(x_content2_train)
    x_content_test = np.array(x_content2_test)
    x_time_train = np.array(x_time2_train_).reshape(len(x_time2_train_), 10)
    x_time_test = np.array(x_time2_test_).reshape(len(x_time2_test_), 10)
    return x_content_train, x_time_train, y_train, x_content_test, x_time_test, y_test


def evaluate(pre_, y_test):
    y_true_ = y_test
    y_pre = []
    y_true = []
    for i in pre_:
        y_pre.append(np.argmax(i))
    for i in y_true_:
        y_true.append(np.argmax(i))

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_true, y_pre)
    print(acc)
    from sklearn.metrics import recall_score

    recall = recall_score(y_true, y_pre, average='macro')
    from sklearn.metrics import f1_score

    f1 = f1_score(y_true, y_pre, average='macro')

    from sklearn.metrics import confusion_matrix

    cnf_matrix = confusion_matrix(y_true, y_pre)
    print(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    print("FP")
    print(FP)
    print("FN")
    print(FN)
    print("TP")
    print(TP)
    print("TN")
    print(TN)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    print(FPR)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print(ACC)


# target list
top = TOP55_SORT[0:10]

x_content_train, x_time_train, y_train, x_content_test, x_time_test, y_test = build_train_test_data(top)

m = build_model(top)

m.fit(x={'char_input': x_content_train, 'input_time': x_time_train}, y=y_train,
      epochs=200, batch_size=128,
      verbose=2)
# a, b = m.evaluate(x={'char_input': x_content_test, 'input_time': x_time_test},
#                   y=y_test, verbose=1)
# print(a, b)
pre_ = m.predict({'char_input': x_content_test, 'input_time': x_time_test})
evaluate(pre_, y_test)
