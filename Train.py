from sklearn.model_selection import train_test_split
from Utils import *
import os
print('Setting Up')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


path = '/Users/sangyy/Documents/beta_simulator_mac/dataset'
data = importData(path)

data = balancedData(data, display=True)

imagesPath, steerings = loadData(path, data)

xTrain, xVal, yTrain, yVal = train_test_split(
    imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

model = createModel()
model.summary()

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10, callbacks=[checkpoint],
                    validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

model.save('model.h5')
print('End!!! ALL Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('figure.png')
