exp_dict = {
    'GPU': False,  # Choose False for colab
    'target': 'w', 'one-hot': True,
    'ftr_folder': './data/ftr', 'time_range': [-0.01, 0.006], 'resolution': {'figsize':(4, 4),'dpi':64}, 'ftype': 'jpeg',
    'sample':'group1', 'test_size': None, 'random_seed': 5,
    'model_name': 'cnn1', 'max_epoch': 350, 'measure': 'acc', 'measure_val': 0.95,
    'show_plot': False
    
}


#===============================================================================
print('Preparing Data...')
start = datetime.datetime.now()

# If using GPU.
if exp_dict['GPU']:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# read database.
f = h5py.File('GWdatabase.h5','r')
# create a list that contains all the failure cases.
fail_list, non_fail_list = list_fail_case(f)


# Create image x
prepare_x_image(f, fail_list, exp_dict['ftr_folder'], exp_dict['time_range'], exp_dict['resolution'], exp_dict['ftype'], overwrite=False)
# Todo: Data augmentation


stop = datetime.datetime.now()
print('Done! Time =', stop - start)
sys.exit()

#===============================================================================
print('Training model...')
start = datetime.datetime.now()

if exp_dict['sample'] == 'group1':
    g_dict = group_dict1

for key, value in g_dict.items():
    train_list = [x for x in non_fail_list if x.split('_')[1] not in value]
    test_list = [x for x in non_fail_list if x.split('_')[1] in value]
    
    # Todo: load data w/ augmentation
    # Create y
    y_train = prepare_y(train_list, exp_dict['target'])
    y_test = prepare_y(test_list, exp_dict['target'])
    num_of_label0 = len(np.unique(y_train))
    num_of_label1 = len(np.unique(y_test))
    if set(np.unique(y_train)) != set(np.unique(y_test)):
      sys.exit('\n For %s in %s, train label != test label.' % (exp_dict['target'], exp_dict['sample']))

    # Load cleaned X data
    X_train = load_x_image(train_list, exp_dict['ftr_folder'], exp_dict['time_range'], exp_dict['resolution'], exp_dict['ftype'])
    X_test = load_x_image(test_list, exp_dict['ftr_folder'], exp_dict['time_range'], exp_dict['resolution'], exp_dict['ftype'])
    
    # transfer the shape of image data for model training
    w = exp_dict['resolution']['figsize'][0]*exp_dict['resolution']['dpi']
    h = exp_dict['resolution']['figsize'][1]*exp_dict['resolution']['dpi']
    X_train = X_train.reshape(X_train.shape[0], w, h, 1)
    X_test = X_test.reshape(X_test.shape[0], w, h, 1)
    #using one hot to encode the label.
    if exp_dict['one-hot']:
        label_encoder0 = LabelEncoder()
        label_encoder0.fit(y_train)
        y_train = label_encode(label_encoder0, y_train)
        label_encoder1 = LabelEncoder()
        label_encoder1.fit(y_test)
        y_test = label_encode(label_encoder1, y_test)

    # Model config
    model = prepare_NN(exp_dict['model_name'], num_of_label0)




    # Train
    epochs = 100
    _history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
    hist_dict = _history.history
    _acc = hist_dict[exp_dict['measure']][-1]


    while all([epochs <= exp_dict['max_epoch'], _acc < exp_dict['measure_val']]):
        if _acc < 0.9:
            _epochs = 50
        elif all([_acc >= 0.9, _acc < 0.95]):
            _epochs = 20
        elif _acc >= 0.95:
            _epochs = 10

        _history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=_epochs, batch_size=32)
        #history = model.fit(x=[X_train, X_fpeak_train], y=y_train, validation_data=([X_test, X_fpeak_test], y_test), epochs=200, batch_size=32)

        for k in hist_dict.keys():
            hist_dict[k] = hist_dict[k] + _history.history[k]

        epochs += _epochs
        _acc = hist_dict[exp_dict['measure']][-1]


    # Evaluate
    print('%s_%s_%s-%s' % (exp_dict['model_name'], exp_dict['target'], exp_dict['sample'], str(key)))
    if exp_dict['one-hot']:
        y_test, y_pred, acc = evaluate_model(y_test, X_test, label_encoder=label_encoder)
        print('test accuracy =', acc)
        #plot confusion matrix
        plt_save_path = './data/plot/cm_%s_%s_%s-%s_acc%s.jpg' % (exp_dict['model_name'], exp_dict['target'], exp_dict['sample'], str(key), str(round(acc, 2)))
        plot_confusion_matrix(labels, y_test, y_pred, plt_save_path)
        if exp_dict['show_plot']:
            plt.show()
    else:
        y_test, y_pred, acc = evaluate_model(y_test, X_test, label_encoder=label_encoder)
        print('test accuracy =', acc)


    # Save model & result
    model_fname = './model/model_%s_%s_%s-%s_acc%s.h5' % (exp_dict['model_name'], exp_dict['target'], exp_dict['sample'], str(key), str(round(acc, 2)))
    model.save(model_fname)
    hist_fname = './data/result/hist_%s_%s_%s-%s_acc%s.txt' % (exp_dict['model_name'], exp_dict['target'], exp_dict['sample'], str(key), str(round(acc, 2)))
    with open(hist_fname, "wb") as file:
        pickle.dump(hist_dict, file=file)

    # Load model
    # model = keras.models.load_model(model_fname)
    # with open(hist_fname, "rb") as file:
    #     d0 = pickle.load(file)



    stop = datetime.datetime.now()
    print('Done! Time =', stop - start)
