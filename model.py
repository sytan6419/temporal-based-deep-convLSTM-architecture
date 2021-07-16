def evaluate_model(trainX, trainy, testX, testy):

    _verbose, epochs, batch_size = 0, 500, 128
    n_timesteps, n_features, n_outputs = SLIDING_WINDOW_LENGTH, SENSOR_CHANNELS, NUM_CLASSES

    #Feature Learning Pipeline 1
    inputs1 = Input(shape=(n_timesteps,n_features), name='x1')
    convs1_1 = Conv1D(filters=8, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(inputs1)
    pools1_1 = MaxPooling1D(pool_size=2, strides=1)(convs1_1)
    drops1_1 = Dropout(0.5)(pools1_1)
    convs1_2 = Conv1D(filters=18, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(drops1_1)
    pools1_2 = MaxPooling1D(pool_size=2, strides=1)(convs1_2)
    drops1_2 = Dropout(0.5)(pools1_2)
    convs1_3 = Conv1D(filters=36, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(drops1_2)
    pools1_3 = MaxPooling1D(pool_size=2, strides=1)(convs1_3)
    drops1_3 = Dropout(0.5)(pools1_3)

    
    #Feature Learning Pipeline 2
    inputs2 = Input(shape=(n_timesteps,n_features), name='x2')
    convs2_1 = Conv1D(filters=8, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(inputs2)
    pools2_1 = MaxPooling1D(pool_size=2, strides=1)(convs2_1)
    drops2_1 = Dropout(0.5)(pools2_1)
    convs2_2 = Conv1D(filters=18, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(drops2_1)
    pools2_2 = MaxPooling1D(pool_size=2, strides=1)(convs2_2)
    drops2_2 = Dropout(0.5)(pools2_2)
    convs2_3 = Conv1D(filters=36, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(drops2_2)
    pools2_3 = MaxPooling1D(pool_size=2, strides=1)(convs2_3)
    drops2_3 = Dropout(0.5)(pools2_3)

    #Feature Learning Pipeline 3
    inputs3 = Input(shape=(n_timesteps,n_features), name='x3')
    convs3_1 = Conv1D(filters=8, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(inputs3)
    pools3_1 = MaxPooling1D(pool_size=2, strides=1)(convs3_1)
    drops3_1 = Dropout(0.5)(pools3_1)
    convs3_2 = Conv1D(filters=18, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(drops3_1)
    pools3_2 = MaxPooling1D(pool_size=2, strides=1)(convs3_2)
    drops3_2 = Dropout(0.5)(pools3_2)
    convs3_3 = Conv1D(filters=36, kernel_size=2, strides=1, kernel_regularizer='l2', padding='same', activation='tanh')(drops3_2)
    pools3_3 = MaxPooling1D(pool_size=2, strides=1)(convs3_3)
    drops3_3 = Dropout(0.5)(pools3_3)

    #concatenate the feature from all Feature Learning Pipeline
    merged = concatenate([drops1_3, drops2_3, drops3_3], axis=1)

    #Sequential Learning
    lstm_1 = LSTM(48, return_sequences=False, kernel_regularizer='l2', activation='tanh')(merged)
    drops3_1 = Dropout(0.5)(lstm_1)

    #Softmax Classifier
    output = Dense(n_outputs, activation='softmax')(drops3_1)
    
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit({'x1':trainX[:-2], 'x2':trainX[1:-1], 'x3':trainX[2:]}, trainy[2:], epochs=epochs, batch_size=batch_size, verbose=_verbose)

    return model
