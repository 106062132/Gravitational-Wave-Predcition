exp_dict = {
    'target': 'EOS', 'one-hot': True,
    'ftr_folder': './ftr', 'time_range': [-0.01, 0.006], 'resolution': {'figsize':(4, 4),'dpi':64}, 'ftype': 'jpeg',
    'test_size': 0.2, 'random_seed': 5,
    'model_name': 'cnn1', 'max_epoch': 350, 'measure': 'acc', 'measure_val': 0.99
    
}


#===============================================================================
print('Preparing Data...')
start = datetime.datetime.now()

# Colab: download data from web
if not os.path.isfile('./GWdatabase.h5'):
    !wget -O GWdatabase.h5 https://zenodo.org/record/201145/files/GWdatabase.h5?download=1
# read database.
f = h5py.File('GWdatabase.h5','r')
# create a list that contains all the failure cases.
fail_num, fail_case, fail_list = list_fail_case(f)


# Create y
labels = prepare_y(f, fail_list, exp_dict['target'])
num_of_label = len(np.unique(labels))
# Create image x
prepare_x_image(f, fail_list, exp_dict['ftr_folder'], exp_dict['time_range'], exp_dict['resolution'], exp_dict['ftype'], overwrite=False)
# Todo: Data augmentation


stop = datetime.datetime.now()
print('Done! Time =', stop - start)


#===============================================================================
print('Training model...')
start = datetime.datetime.now()

# Load cleaned data
# Todo: load data w/ augmentation
data = load_x_image(f, fail_list, exp_dict['ftr_folder'], exp_dict['time_range'], exp_dict['resolution'], exp_dict['ftype'])


# Training config
# split the data
X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=exp_dict['test_size'],random_state=5)
# transfer the shape of image data for model training
w = exp_dict['resolution']['figsize'][0]*exp_dict['resolution']['dpi']
h = exp_dict['resolution']['figsize'][1]*exp_dict['resolution']['dpi']
X_train = X_train.reshape(X_train.shape[0], w, h, 1)
X_test = X_test.reshape(X_test.shape[0], w, h, 1)
#using one hot to encode the label.
if exp_dict['one-hot']:
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encode(label_encoder, y_train)
    y_test = label_encode(label_encoder, y_test)


# Model config
model = prepare_NN(exp_dict['model_name'], num_of_label)




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




# Save model & result
model_fname = 'model_%s_%s_acc%s.h5' % (exp_dict['model_name'], exp_dict['target'], str(round(acc, 2)))
model.save(model_fname)
hist_fname = 'hist_%s_%s_acc%s.txt' % (exp_dict['model_name'], exp_dict['target'], str(round(acc, 2)))
with open(hist_fname, "wb") as file:
    pickle.dump(hist_dict, file=file)

# Load model
# model = keras.models.load_model(model_fname)
# with open(hist_fname, "rb") as file:
#     d0 = pickle.load(file)


# Evaluate
if exp_dict['one-hot']:
    y_test, y_pred, acc = evaluate_model(y_test, X_test, label_encoder=label_encoder)
    print('test accuracy =', acc)
    #plot confusion matrix
    plot_confusion_matrix(labels, y_test, y_pred)
    plt.show()
else:
    y_test, y_pred, acc = evaluate_model(y_test, X_test, label_encoder=label_encoder)
    print('test accuracy =', acc)




stop = datetime.datetime.now()
print('Done! Time =', stop - start)
