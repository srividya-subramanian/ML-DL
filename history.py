error.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
w.shape
b.shape
y_pred
y_pred.shape
y_train.shape
err = (y_pred - y)
err = (y_pred - y_train)
err.shape
X_train.shape
np.dot(X_train,err)
np.dot(X_train,err.T)
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
err = (y_pred - y).transpose()
err
err.shape
err.T.shape
np.dot(X_train,err)
X_train.shape, err.shape
error.shape, images.shape
np.dot(images.T, error)
np.dot(X_train,err.T)
np.dot(X_train.T,err)
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
X_train.shape, err.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
neuron_idx
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled5.py --wdir
visualize_weights(trained_weights, 2)  # Neuron für die Ziffer "2"
%runfile C:/Users/srivi/.spyder-py3/untitled5.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
visualize_weights(w, 1)  # Neuron für die Ziffer "2"
visualize_weights(w, 0)  # Neuron für die Ziffer "2"
visualize_weights(w, 3)  # Neuron für die Ziffer "2"
visualize_weights(w, 4)  # Neuron für die Ziffer "2"
visualize_weights(w, 5)  # Neuron für die Ziffer "2"
%runfile C:/Users/srivi/.spyder-py3/untitled5.py --wdir
visualize_weights(w, 0)  # Neuron für die Ziffer "2"
visualize_weights(w, 1)  # Neuron für die Ziffer "2"
visualize_weights(trained_weights, 2)  # Neuron für die Ziffer "2"
visualize_weights(w, 3)  # Neuron für die Ziffer "2"
visualize_weights(w, 4)  # Neuron für die Ziffer "2"
visualize_weights(w, 5)  # Neuron für die Ziffer "2"
visualize_weights(w, 9)  # Neuron für die Ziffer "2"
visualize_weights(w, 8)  # Neuron für die Ziffer "2"
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
w
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled8.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled9.py --wdir
from chatterbot import ChatBot
pip install chatterbot
pip install SpeechRecognition
install gTTS
pip install gTTS
pip install transformers
%runfile C:/Users/srivi/.spyder-py3/untitled9.py --wdir
import pytorch
pip install pytorch
pip install Pyrebase4
pip install pytorch
pip install pytorch --no-binary :all:
    
pip install pytorch --no-binary :all
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
pip install tflearn
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tflearn
pip uninstall tflearn
pip install git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing
pip install git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
%runfile C:/Users/srivi/.spyder-py3/chatbot_NN.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
intents
intents['tag']
intent['tag']
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
training
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
bag
pattern_words
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
training.shape
training.to_array()
training
np.array(training)
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
st
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
st
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
training
training[:,0]
training[0,:]
bag
output_row
pattern_words
words
classes
output_row
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
training
training.shape
len(training)
len(training[0])
len(training[1])
len(training[2])
len(training)
random.shuffle(training)
np.array(training)
training = np.array(training, dtype="object")
training.shape
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
pip install git+https://github.com/tflearn/tflearn.git
len(train_y[0]
)
len(train_x[0]
)
training.shape
list(training[:,0])
len(list(training[:,0]))
len(list(training[:,1))
len(list(training[:,1]))
input_shape = [train_x.shape[1]]
input_shape =[train_x.shape[1]]
train_x.shape[1]
len(train_x)
input_shape = [training.shape[1]]
input_shape
len(train_x[0])
len(train_y[0])
(training[:,0])
(training[:,0]).shape
train_x = list(training[:,0])
len(train_x[0])
len(train_y[0])
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
len(train_x)
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
train_x.shape
train_x
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
train_x.shape
train_x[0]
train_x[1]
train_x[26]
len(list(train_x[26]))
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
train_x
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
train_y
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
tx.shape
len(tx)
len(ty)
len(train_x)
len(train_y)
tx[0]
ty[0]
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
tf.enable_eager_execution()
print(tf.__version__)
tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()
%runfile C:/Users/srivi/.spyder-py3/untitled11.py --wdir
prediction = nn.predict_scores(X_val[0])[0]    
print ("Scores")
print (prediction)    
np.argmax(prediction)

predict_class = nn.predict(X_val[0])[0]
predict_class    

y_val[0]
X_val
i=1    
prediction = nn.predict_scores(X_val[i])[0]    
print ("Scores")
print (prediction)    
np.argmax(prediction)

predict_class = nn.predict(X_val[i])[0]
predict_class    

y_val[0]
y_val[i]
i=5    
prediction = nn.predict_scores(X_val[i])[0]    
print ("Scores")
print (prediction)    
np.argmax(prediction)

predict_class = nn.predict(X_val[i])[0]
predict_class    

y_val[i]
%runfile C:/Users/srivi/.spyder-py3/untitled8.py --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_Activation_Functions_NN.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
print(rf.predict(p))
p
p = bow("is your shop open today?", words)
p
print(p)
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
print (p)
words
len(words)
enumerate(words)
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
print(rf.predict(p))
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
p
len(p)
train_x
train_x[0]
train_x[0].shape
len(train_x[0])
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
p
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
print(rf.predict(list(p)))
sentence = "is your shop open today?"
p = bow(sentence, words)
test = np.array(p, dtype="object")
test = list(test)
#p = p.reshape(-1, 1))
print (p)
#print (classes)

print(rf.predict(test))
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
print(rf.predict(test))
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
%runfile C:/Users/srivi/.spyder-py3/chatbot_NN.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled12.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled13.py --wdir
one_hot_labels
one_hot_labels.shape
one_hot_labels[10,*]
one_hot_labels(10,*)
one_hot_labels(10,:)
one_hot_labels[10,:]
one_hot_labels[0:10,:]
labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:1000].reshape(1000,28*28)/255, y_train[0:1000])
labels
one_hot_labels = np.zeros((len(labels),10))
one_hot_labels[0:10,:]
enumerate(labels)
len(images)
len(images)/batch_size
i=1
(i * batch_size),((i+1)*batch_size)
images.shape
layer_0.shape
dropout_mask
layer_0 = images[batch_start:batch_end]
layer_1 = perceptron(layer_0,weights_0_1)
layer_1 = relu(layer_1)
dropout_mask = np.random.randint(2,size=layer_1.shape)
def perceptron(x, w): 
    return np.dot(x, w)
    
def perceptron(x, w): 
    
return np.dot(x, w)
    
def perceptron(x, w): 
    return np.dot(x, w)
    
layer_0 = images[batch_start:batch_end]
layer_1 = perceptron(layer_0,weights_0_1)
layer_1 = relu(layer_1)
dropout_mask = np.random.randint(2,size=layer_1.shape)
dropout_mask
layer_1
layer_1 *= dropout_mask * 2
layer_1
layer_2 = perceptron(layer_1,weights_1_2)
correct_cnt
%runfile C:/Users/srivi/.spyder-py3/untitled13.py --wdir
labels_0, layer_2
labels_0 - layer_2
(labels_0 - layer_2).round(3)
(labels_0 - layer_2).round(1)
(labels_0 - layer_2).round(2)
(labels_0 - layer_2).round(3)
(labels_0 - layer_2).round(2)
%runfile C:/Users/srivi/.spyder-py3/untitled13.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
data
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
i=0,l=1
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
i=0,l=1
i=0l=1
i=0
l=1
data[i][l]
data
data[i], data[l]
data[i,l]
data(i,l)
data[i][l]
i
l
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
rows
cols
data
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
data
data.shape
data[0,1]
data[0,0]
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
i
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
ii
jj
jj.int
int(ii)
data_new = [ii,jj]
data_new
np.array(data_new)
int(data_new)
data_new.int
data_new.astype(int)
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
data_new
data_new.shape
data_new.T
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
data_new.T
data_new
data_new.shape
labels
labels.astype(int)
labels.astype(int).T
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
ypred
y_pred
w1
w2
x
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
terr
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
terr
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled15.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled15.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled15.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled16.py --wdir
x.shape, deltay.shape
X
x.shape
%runfile C:/Users/srivi/.spyder-py3/untitled16.py --wdir
x.shape
X
x
x * deltay
deltay
y
y_pred
x * deltay * alpha
%runfile C:/Users/srivi/.spyder-py3/untitled16.py --wdir
w
%runfile C:/Users/srivi/.spyder-py3/untitled16.py --wdir
terr
%runfile C:/Users/srivi/.spyder-py3/untitled16.py --wdir
terr
%runfile C:/Users/srivi/.spyder-py3/untitled16.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled13.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled9.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled19.py --wdir
test_correct_cnt
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
data
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
speed = pd.DataFrame(rand.uniform(0,200,1000))
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
df
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
df
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
X_train.shape, y_train.shape, X_test.shape, y_test.shape
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
X_train.shape, y_train.shape, X_test.shape, y_test.shape
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
y_pred.shape, y_trian.shape
y_pred.shape, y_train.shape
y_pred.reshape(1)
y_pred.reshape(-1)
(y_pred.reshape(-1)).shape
y_pred.shape, y_train.shape
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
err
err.shape
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
err
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
w0.shape
wo.shape
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
err.shape
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
err.shape
wo.shape
derivative_sigmoid_fn(y_pred)
derivative_sigmoid_fn(y_pred).shape
%runfile C:/Users/srivi/.spyder-py3/untitled23.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
data
data.T
np.array(data)
np.array(data).T
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
data
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
y
y.shape
data.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
X.shape, y.shape
X
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
y_pred
y_pred.shape
y.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
err.shape
err
ypred.shape
y_pred.shape, y.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
err
err.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
deltao.shape, wo.T.shape, wo.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
x.shape
X.shape
wi.shape
np.dot(X,wi).shpae
np.dot(X,wi).shape
np.dot(X,wi)+bi.shape
bi.shape
(np.dot(X,wi)+bi).shape
hx = perceptron_npdot(x,wi)+bi
h  = sigmoid_activationfn(hx)
hy = perceptron_npdot(h,wo)+bo
y_pred = (sigmoid_activationfn(hy)).reshape(-1)
h.shape
hx.shaüe
hx.shape
x=X
hx = perceptron_npdot(x,wi)+bi
h  = sigmoid_activationfn(hx)
hy = perceptron_npdot(h,wo)+bo
y_pred = (sigmoid_activationfn(hy)).reshape(-1)
hx.shape, h.shape, hy.shape, y_pred.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled25.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
d_predicted_output.shape
weights_hidden_output.shape
d_predicted_output.dot(weights_hidden_output.T).shape
err.shape
deltao.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
err.shape
y.shape
np.array([[0], [1], [0], [1], [0], [1]]).shape
np.array(Art).T.shape
np.array([[5.1, 3.5, 1.4, 0.2], 
              [7.0, 3.2, 4.7, 1.4], 
              [4.6, 3.1, 1.5, 0.2], 
              [6.5, 2.8, 4.6, 1.5], 
              [5.0, 3.6, 1.4, 0.2], 
              [5.7, 2.8, 4.5, 1.3]]).shape
X.shape
np.array(Art).T.reshape(1)
np.array(Art).T.reshape(1,1)
np.array(Art).T.reshape(6,1)
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled25.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
[y.round(2), y_pred.round(1)]
%runfile 'C:/Users/srivi/.spyder-py3/DL_Übung76ü01_NN_FPBP_myworkingversion.py' --wdir
y_pred
%runfile 'C:/Users/srivi/.spyder-py3/DL_Übung76ü01_NN_FPBP_myworkingversion.py' --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
y_train.shape
y_pred.shape
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
y_pred.shape
y_train.shape
y_train
y_pred
%runfile C:/Users/srivi/.spyder-py3/untitled21.py --wdir
import tensorflow.data.Dataset
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
pip install tensorflow_datasets
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tf.keras.preprocessing.image.ImageDataGenerator
import tensorflow as tf
import tf.keras.preprocessing.image.ImageDataGenerator
import tensorflow.keras.preprocessing.image.ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
import glob
import glob
data = glob.glob(dataset_path)
dataset_path = "Users/srivi/Documents/ML_data/cat_dog/training_set/cats/*.jpg"
import glob
data = glob.glob(dataset_path)
data
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
images
images.value
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
images
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
plot_image_grid(cat_files[:16])
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
images = [tf_keras.preprocessing.image.load_img(img) for img in train_cat_files]
plot_image_grid(train_cat_files[0:16])
images
images.shape
images[0]
images[0].shape
len(images[0])
plt.imshow(images[0])
plt.show()
plt.imshow(images[1])
plt.show()
plt.imshow(images[12)
plt.imshow(images[12])
plt.show()
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
train_cat_files
train_cat_data = [tf_keras.preprocessing.image.load_img(img) for img in train_cat_files]
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
y_train
train_cat_data[0]
train_cat_data[0].flatten()
print("train size: {} cats and {} dogs".format(len(train_cat_data), len(train_dog_data)))
print("test size :  {} cats and  {} dogs".format(len(test_cat_data), len(test_dog_data)))'
print("train size: {} cats and {} dogs".format(len(train_cat_data), len(train_dog_data)))
print("test size :  {} cats and  {} dogs".format(len(test_cat_data), len(test_dog_data)))
train_cat_data = train_cat_data[0:3999]
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
train_cat_dataset_path
cd train_cat_dataset_path
training_set = train_datagen.flow_from_directory(train_cat_dataset_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
train_generator
train_generator.samples
train_generator.data
train_generator[0]
train_generator[0].shape
len(train_generator[0]=
len(train_generator[0])
train_generator[0][0]
train_generator[0][0].shape
train_generator[0][1].shape
train_generator = train_datagen.flow_from_directory(train_dataset_path, target_size = IMG_SIZE,
                                                   # classes=['Cat' , 'Dog'],
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    #save_to_dir=aug_data_path,
                                                    #save_prefix='aug_',
                                                    #save_format="jpg",
                                                    seed = 1
                                                    )
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
train_generator[0][0].shape
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
train_generator[0][0].shape
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Sequential,Conv2D, MaxPooling2D,Flatten,Dense
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
test_dog_data.shape
len(test_dog_data)
test_dog_data[0]
xx=test_dog_data[0]
xx.shape
import os, cv2, glob
import opencv as cv2
pip install opencv
import opencv
import cv2
pip install opencv-python
import opencv
import cv2
img = cv2.imread(train_generator[0])
from skimage.io import imread, imshow
img = imread(train_generator[0])
img = imread(train_generator)
test_dog_data[0]
test_dog_data[0].shaper
test_dog_data[0].shape
from PIL import Image
img = process_image(Image.open(test_dog_data[0]))
from numpy import asarray
asarray(test_dog_data[0])
asarray(test_dog_data).shape
asarray(test_dog_data).shape()
xx=asarray(test_dog_data)
xx=asarray(test_dog_data[0])
xx.shape
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
df
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
df
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
df2
df1
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
df1
df2
df = [df1,df2]
df
df = [df1,df2,axis=0]
df_m =df1.append(df2, ignore_index=True)
df_m = pd.concat([df1, df2], ignore_index=True, sort=False)
df_m
df_m.shape
df_m.drop([df.index[8006]])
df_m.drop([8006])
df_m = pd.concat([df1, df2], ignore_index=True, sort=False).drop([8006]).sample(frac = 1)
df_m.shape
df_m
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
df_m = pd.concat([df1, df2], ignore_index=True, sort=False).drop([8006]).sample(frac = 1).reset_index()
df_m
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
print(dogs[:5])
from torchvision import transforms
pip install torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
transform = transforms.Compose([transforms.Resize((50,50)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
dataset = ImageFolder(DATA_DIR+'/training_set', transform=transform)
test_dataset = ImageFolder(DATA_DIR+'/test_set', transform=transform)
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
dataset
dataset.shape
dataset[0]
dataset[0].shape
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
img, label = dataset[0]
print(img.shape, label)
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
len(dataset)
from torch.utils.data import Dataset, DataLoader
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
(train_features, train_targets), (test_features, test_targets) = mnist.load_data()
train_features.shape
train_loader
train_loader[0]
train_loader[0].shape
dataset[0].shape
print(img.shape, label)
print(img.shape)
dataset[0]
np.array(dataset[0])
np.array(img)
np.array(img).shape
np.array(img).flatten().shape
img2 = Image.fromarray(img.reshape(200,300,3), 'RGB')
img.shape
img2 = Image.fromarray(img.reshape(28,28,3), 'RGB')
import tensorflow as tf
from tensorflow import keras

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train
images_train.shape
dataset
dataset[0]
dataset[0].shape
train_generator.shape
train_generator[0].shape
train_cat_data.shape
train_cat_data[0].shape
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled29.py --wdir
112/4
%runfile C:/Users/srivi/.spyder-py3/untitled29.py --wdir
print(photos.shape, labels.shape)
photo
photo.shape
if photo.shape[-1] == 3: print(photo.shape)
image = skimage.color.rgb2gray(photo)
import skimage
image = skimage.color.rgb2gray(photo)
image.shape
plt.plot, image
show_example(*image)
def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img)
    #plt.imshow(img.permute(1, 2, 0))
    plt.show()
    
show_example(*image)
show_example(image)
image.imshow(image)
pixel =image
greyscale = [pixel if pixel > 120 else 0 for pixel in pixel]
bildArray = np.array(pixel, dtype=int).reshape((28, 28)) #7

plt.imshow(bildArray, cmap='Greys') #8
plt.show() #9
max(pixel)
image.shape
plt.imshow(image, cmap='Greys') #8
plt.show() #9
minmax(image)
arr = image
numpy.max(arr)
np.max(arr)
np.min(arr)
save('dogs_vs_cats_photos.npy', photos)
save('dogs_vs_cats_labels.npy', labels)
%runfile C:/Users/srivi/.spyder-py3/untitled29.py --wdir
photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
photos.shaüe
photos.shape
file
%runfile C:/Users/srivi/.spyder-py3/untitled29.py --wdir
photos.shape
photos.reshape(-1)
photos.reshape(25000, 28, 28,-1)
photo.reshape(25000, 28, 28,-1)
photos.reshape(25000, 28, 28,-1)
xx=photos.reshape(25000, 28, 28,-1)
xx.shape
xx=photos.reshape(25000, 28, 28)
xx.shape
photo.shape
greyphoto.shape
img_to_array(greyphoto).shape
greyphoto
%runfile C:/Users/srivi/.spyder-py3/untitled29.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled30.py --wdir
labels.shape
%runfile C:/Users/srivi/.spyder-py3/untitled30.py --wdir
labels
photos.shape#
(train_features, train_targets), (test_features, test_targets) = mnist.load_data()
train_features.shape
%runfile C:/Users/srivi/.spyder-py3/untitled30.py --wdir
photos.shape#
labels.shape
%runfile C:/Users/srivi/.spyder-py3/untitled30.py --wdir
hx.shape
x.shape
wi.shape
%runfile C:/Users/srivi/.spyder-py3/untitled30.py --wdir
bi.shape
xx=perceptron_npdot(x,wi)+bi
hx.shape
ho.shape
perceptron_npdot(x,wi)
x.shape
%runfile C:/Users/srivi/.spyder-py3/untitled30.py --wdir
hx.shape
print(perceptron_npdot(x,wi))+bi
%runfile C:/Users/srivi/.spyder-py3/untitled30.py --wdir
X.shape
y.shape
%runfile 'C:/Users/srivi/.spyder-py3/DL_teilprüfung2.py' --wdir

## ---(Wed Feb 12 09:15:36 2025)---
%runfile 'C:/Users/srivi/.spyder-py3/DL_teilprüfung2.py' --wdir
hx.shape
X.shape
wi.shape
x=X
hx=perceptron_npdot(x,wi))+bi)
hx=perceptron_npdot(x,wi))+bi
hx=perceptron_npdot(x,wi)+bi
ho  = relu_activationfn(hx)
hx.shape
np.min(hx)
np.min(wi)
wi
hx
hx.shape
relu_activationfn(hx)
sigmoid_activationfn(hx)
%runfile C:/Users/srivi/.spyder-py3/untitled0.py --wdir
data
%runfile C:/Users/srivi/.spyder-py3/untitled0.py --wdir
data.shape
data.data.shape
%runfile C:/Users/srivi/.spyder-py3/untitled0.py --wdir
digits.data
digits.data.shape
digits.images
digits.images.shape
digits.labels
digits.target
digits.targets
digits.target
%runfile C:/Users/srivi/.spyder-py3/untitled0.py --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
X.shape
data.shape
images.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
X.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
preds.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
preds.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new.shape
data_new[0]
data_new[0].shape
data_new[1].shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
X_new..reshape(8, 8)
X_new.reshape(8, 8)
X_new.reshape(100,8, 8)
xx=X_new.reshape(100,8, 8)
xx.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
labels
labels.drop['0']
labels.drop[0,axis=0]
labels.shape
np.delete(labels, np.where(
    (labels = 5))[0], axis=0)
np.delete(labels, np.where(
    (labels == 5))[0], axis=0)
np.delete(labels, np.where((labels == 0))[0], axis=0)
data(np.delete(labels, np.where((labels == 0))[0], axis=0))
np.delete(data, np.where((labels == 0))[0], axis=0)
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
np.(data, np.where((labels == 0))[0], axis=0)
(data, np.where((labels == 0))[0], axis=0)
data, np.where((labels == 0))[0]
data, np.where((labels == 0))
labels, np.where((labels == 0))
labels(np.where(labels == 0))
labels[np.where(labels == 0)]
data[np.where(labels == 0)]
labels[np.where(labels == 1)]
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
labels[np.where(labels == 1)]
labels[np.where(labels == 2)]
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
label
label[0]
data[0]
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
X_new.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
label1.shape
data1.shape
y_new.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new
data_new.shape
len(data_new)
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
len(data_new)
len(data_new[0])
data_new.shape
data_new.shae
len(data_new[010])
len(data_new[10])
len(data_new[100])
len(data_new[677])
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
print(data[0].shape)
print(data[0].[0]
print(data[0][0]
)
data[0][0].shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
i=0
j=1
data[i][j]
data[i][j].shape
data[i][j].shape.reshape(8, 8)
data[i][j].reshape(8, 8)
data[i][j].reshape(8, 8)..reshape(8, 8)
data[i][j].reshape(8, 8).shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
np.array(data_new).shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
new[0].shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new = np.array([])
data_new.shape
X_new
X_new.shape
data_new.append(X_new)
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new.shape
X_new.shape
data_new=X_new
data_new=np.append(data_new, X_new)
data_new.shape
data_new=X_new
data_new.shape
data_new=X_new
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new.shape
X_new
X_new.shape
data_new = X_new
data_new=np.concatenate(data_new, X_new)
data_new=np.concatenate((data_new, X_new),axis=0)
data_new.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new.shape
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
data_new.shape
i=0,k=0
i=0k=0
i=0
k=0
ax[i][k].imshow(data_new[i+k*10],cmap='gray')
ax[i][k].imshow(data_new[i+k*10].reshape(8,8),cmap='gray')
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
plt.show()
precision_score(y, y_pred)
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
enumerate(ax.flat)
%runfile C:/Users/srivi/.spyder-py3/DL_basic_GenAI_GaussianMixtureModel_MNIST.py --wdir
%runfile 'C:/Users/srivi/.spyder-py3/DL_teilprüfung2.py' --wdir
print(accuracy)#, precision, recall, f1score)
accuracy = accuracy_score(y, y_pred)
print(accuracy)#, precision, recall, f1score)
y_pred
%runfile 'C:/Users/srivi/.spyder-py3/DL_teilprüfung2.py' --wdir
wo
wi, wo, bi, bo = initialise()
wi
X = photos.reshape(25000,3136)
y = labels.reshape(25000,1)

def initialise():
    ipLayerSize, hdLayerSize, opLayerSize = 3136,32,1
    wi = np.random.rand(ipLayerSize, hdLayerSize) 
    bi = np.zeros((1, hdLayerSize))
    wo = np.random.rand(hdLayerSize, opLayerSize) 
    bo = np.zeros((1, opLayerSize))
    print(wi.shape, wo.shape, bi.shape, bo.shape)
    return wi, wo, bi, bo
    
wi.shape
hx = perceptron_npdot(x,wi)+bi
ho  = relu_activationfn(hx)
yh = perceptron_npdot(ho,wo)+bo
y_pred = (softmax_activationfn(yh))
y_pred = softmax_activationfn(yh)
%runfile 'C:/Users/srivi/.spyder-py3/DL_teilprüfung2.py' --wdir
y_pred
y
y_pred_2
y_pred_2.shape
wi = np.random.rand(3136,1) 
bi = np.zeros((1, 1))
y_pred_2 = forward_propagation_2(X,wi,bi)
%runfile 'C:/Users/srivi/.spyder-py3/DL_teilprüfung2.py' --wdir

## ---(Thu Feb 13 05:37:25 2025)---
%runfile 'C:/Users/srivi/.spyder-py3/DL_teilprüfung2.py' --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
tokens
tokens.shape
len(token)
len(tokens)
input_dataset
target_dataset
%runfile C:/Users/srivi/.spyder-py3/Project_mnist.py --wdir
error = error(Y_train, y_pred)
print(Y_train[0:20], y_pred[0:20])
yy, h = forward_propagation(X_test, wi, wo, bi, bo)
Y_train
Y_train.shape
y_pred.shape
y_pred[0,:]
%runfile C:/Users/srivi/.spyder-py3/DL_MNIST_NN_SwamiKannanNeural-Network-from-scratch-Numpy_GitHub.py --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_MNIST_CNN_bookversion.py --wdir
layer_2
layer_2.shape
y_pred
layer_2
%runfile C:/Users/srivi/.spyder-py3/Project_mnist.py --wdir
Y_train.shape
%runfile C:/Users/srivi/.spyder-py3/Project_mnist.py --wdir
x.shape
%runfile C:/Users/srivi/.spyder-py3/Project_mnist.py --wdir
error(y, y_pred).shape
y.shape
y_pred.shape
y_batch.shape
y_train_shuffled.shape
Y_train_onehot.shape
Y_train.shape
%runfile C:/Users/srivi/.spyder-py3/Project_mnist.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled3.py --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_image_array_conversion_NN.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
tokenized_docs
vocab
vocab = {word for doc in tokenized_docs for word in doc}
vocab
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
vocab
tokens
tokenized_docs
bow_vectors
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
all words
all_words
all_words1 = [word for review in documents for word in tokenize(review)]
all_words1
all_words1.shape
len(all_words1), len(aa_words)
len(all_words1), len(all_words)
all_words1
all_words
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
len(all_words1), len(all_words)
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
len(all_words1), len(all_words)
vocab
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
y.shape
y_train.shape
X_train.shape
y_test.shape
y_pred.shape
y_train
y_pred
y_pred.round(1)
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
wi.shape
X_train.shape
X_test.shape
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
print(y_test)
print(y_pred)
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
print(y_test)
print(y_pred)
%runfile C:/Users/srivi/.spyder-py3/untitled10.py --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_TF-IDF_NLP.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled12.py --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_RNN_spam_nonspam_chatgpt.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled13.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled14.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled15.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled16.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled17.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled15.py --wdir
z
%runfile C:/Users/srivi/.spyder-py3/untitled22.py --wdir
df
%runfile C:/Users/srivi/.spyder-py3/untitled22.py --wdir
df.shape
df
%runfile C:/Users/srivi/.spyder-py3/untitled22.py --wdir
df
%runfile C:/Users/srivi/.spyder-py3/untitled22.py --wdir
review.shape
%runfile C:/Users/srivi/.spyder-py3/untitled22.py --wdir
data.shape
data
len(data)
len(label)
documents
%runfile C:/Users/srivi/.spyder-py3/untitled22.py --wdir
document
documents
labels
ratings
tokenized_docs
documents[0]
documents[10]
import spacy
import numpy as np

# Load spaCy's pre-trained word embeddings
nlp = spacy.load("en_core_web_md")  # Medium model with word vectors
pip install spacy
import spacy

import numpy as np

# Load spaCy's pre-trained word embeddings
nlp = spacy.load("en_core_web_md")  # Medium model with word vectors
documents
%runfile C:/Users/srivi/.spyder-py3/untitled23.py --wdir
vocabulary = ["amazing", "boring", "exciting", "awful", "great"] 
word_embeddings = generate_random_word_embeddings(vocabulary)
vvocabulary
vocabulary
documents
%runfile C:/Users/srivi/.spyder-py3/untitled23.py --wdir
vocabulary
word_embeddings = generate_random_word_embeddings(vocabulary)
word_embeddings
sentences = ["The movie was amazing and exciting", "The movie was boring and awful"] 
labels = [5, 1, 3, 5, 1, 3, 5, 1]  # 1 für positiv, 0 für negativ 

training_data = np.array([apply_identity_matrix(average_word_vectors(sentence, word_embeddings)) for sentence in sentences]) 
training_labels = np.array(labels)
%runfile C:/Users/srivi/.spyder-py3/untitled23.py --wdir
word_vector
%runfile C:/Users/srivi/.spyder-py3/untitled23.py --wdir
sentence
avg_word_vector
avg_word_vector = average_word_vectors(sentences, word_embeddings)
%runfile C:/Users/srivi/.spyder-py3/DL_movie_review_rating.py --wdir
bow_vectors
vocabulary = ["amazing", "boring", "exciting", "awful", "great", "worst", "best", 'enjoyable', 'decent'] 
word_embeddings = generate_random_word_embeddings(vocabulary)
word_embeddings
word_embeddings.shape
%runfile C:/Users/srivi/.spyder-py3/untitled24.py --wdir
word_embeddings
%runfile 'C:/Users/srivi/.spyder-py3/DL_übung129A01_simpleRNN_NLP.py' --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_autograd_tensorflow.py --wdir
tap = tf.GradientTape()
y = x**2 + 3*x + 5
print(tape.gradient(y, x))
print(tap.gradient(y, x))
with tf.GradientTape() as tape:
    y = x**2 + 3*x + 5  # Function: f(x) = x^2 + 3x + 5

# Compute dy/dx
grad = tape.gradient(y, x)
%runfile 'C:/Users/srivi/.spyder-py3/DL_übung129A01_simpleRNN_NLP.py' --wdir
print("Predicted :", predictions)
%runfile C:/Users/srivi/.spyder-py3/untitled27.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled28.py --wdir
data.shape
len(data)
data
batch_size = 32
time_steps = 50
input_dim = 1
epochs = 100
freq = np.random.uniform(0.1, 1.0, (batch_size, 1))  # Random frequency
phase = np.random.uniform(0, 2 * np.pi, (batch_size, 1))  # Random phase
time = np.linspace(0, 10, time_steps)  # Time steps
data = np.sin(freq * time + phase)  # Sinusoidal pattern
data.shape
def generate_time_series(batch_size, time_steps):
    freq = np.random.uniform(0.1, 1.0, (batch_size, 1))  # Random frequency
    phase = np.random.uniform(0, 2 * np.pi, (batch_size, 1))  # Random phase
    time = np.linspace(0, 10, time_steps)  # Time steps
    data = np.sin(freq * time + phase)  # Sinusoidal pattern
    return np.expand_dims(data, axis=-1)  # Shape: (batch_size, time_steps, 1)
    
X_train = generate_time_series(batch_size, time_steps)
y_train = generate_time_series(batch_size, time_steps)  # Target (can be shifted for forecasting)
y_train.shape
x_train.shape
X_train.shape
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
t1
t1.data
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
word_embeddings = {word: np.zeros(3) for word in vocab}
word_embeddings
vocab = ["Apple","Banana","Orange","Mango","Pineapple"]
print({word: i for i, word in enumerate(sorted(all_words))})
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
embeddings
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
transition_matrix
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
sent2output
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
words
words =  tokenize(sentence)
words
print(sum(word_embeddings[word] for word in words if word in word_embeddings))
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
word_embeddings
words
word_embeddings[word]
vocab
%runfile C:/Users/srivi/.spyder-py3/untitled33.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
np.zeros(len(vocab))
target = np.zeros(len(vocab))
    target[word_to_idx[actual_next_word]] = 1
target = np.zeros(len(vocab))
target[word_to_idx[actual_next_word]] = 1  # Set actual word to 1
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
sum(word_embeddings[word] for word in words if word in word_embeddings)
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
vocab = ["blue", "cloudy", "sunny", "rainy", "sky"]
word_to_idx = {word: i for i, word in enumerate(vocab)}  # Word to index mapping
target = np.zeros(len(vocab))
    target[word_to_idx[actual_next_word]] = 1  # Set actual word to 1
target[word_to_idx[actual_next_word]] = 1
target = np.zeros(len(vocab))
target[word_to_idx[actual_next_word]] = 1
vocan
vocab
target = np.zeros(len(vocab))
target
word_to_idx
target[word_to_idx[actual_next_word]] = 1
target[word_to_idx[actual_next_word]] 
word_to_idx[actual_next_word]
vocab[actual_next_word]
vocab
sent2output
row_vector = np.array([[1, 3, 2]])
col_vector = np.array([[2], [0], [1]])
row_vector.dot(col_vector)
col_vector.dot(row_vector)
np.dot(row_vector, col_vector)
np.dot(col_vector, row_vector)
np.outer(col_vector, row_vector)
np.outer(row_vector, col_vector)
target = np.zeros(len(vocab))
   target[word_to_idx[actual_next_word]] = 1
target = np.zeros(len(vocab))

target[word_to_idx[actual_next_word]] = 1
target = np.zeros(len(vocab))

target[word_to_idx(actual_next_word)] = 1
word_to_idx
target = np.zeros(len(vocab))
target[word_to_idx["sun"]] = 1
target = np.zeros(len(vocab))
target[word_to_idx["blue"]] = 1
target
for i in word_to_idx:
    if i == actual_next_word:
       target[word_to_idx(actual_next_word)] = 1 
    else:
        target = target
        
target
word_to_idx = {word: i for i, word in enumerate(vocab)}  # Word to index mapping
for i in word_to_idx:

    if i == actual_next_word:
       target[word_to_idx(actual_next_word)] = 1 
    else:
        target = target
        
target
word_to_idx = {word: i for i, word in enumerate(vocab)}  # Word to index mapping
target = np.zeros(len(vocab))
target
for i in word_to_idx:

    if i == actual_next_word:
       target[word_to_idx(actual_next_word)] = 1 
    else:
        target = target
        
target
%runfile C:/Users/srivi/.spyder-py3/untitled32.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled36.py --wdir
transition_matrix
sent2output
%runfile c:/users/srivi/.spyder-py3/untitled38.py --wdir
text = open(path)
%runfile c:/users/srivi/.spyder-py3/untitled38.py --wdir
text
%runfile c:/users/srivi/.spyder-py3/untitled38.py --wdir
lines
lines.shape
len(lines)
%runfile c:/users/srivi/.spyder-py3/untitled38.py --wdir
data
len(lines)
%runfile c:/users/srivi/.spyder-py3/untitled38.py --wdir
%runfile c:/users/srivi/.spyder-py3/untitled40.py --wdir
%runfile c:/users/srivi/.spyder-py3/untitled41.py --wdir
sequence_length, _ = X.shape
sequence_length, _
X.shape
sequence_length
_
h_t = np.zeros((self.hsize, 1))  # Initial hidden state
c_t = np.zeros((self.hsize, 1))  # Initial cell state
h_t = np.zeros((hsize, 1))  # Initial hidden state
c_t = np.zeros((hsize, 1))  # Initial cell state
t = 1
x_t = X[t].reshape(-1, 1)  # Get current timestep input

# Concatenate hidden state and input
concat = np.vstack((h_t, x_t))
concat.shape
h_t.shape, x_t.shape
%runfile C:/Users/srivi/.spyder-py3/untitled43.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled44.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled46.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled47.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled48.py --wdir
%runfile C:/Users/srivi/.spyder-py3/DL_CaseStudy1317C01.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled50.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled51.py --wdir
print(y_pred)
print(targets[i])
print(y_pred.data)
%runfile C:/Users/srivi/.spyder-py3/untitled51.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled50.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled52.py --wdir
vocab
%runfile C:/Users/srivi/.spyder-py3/untitled52.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled53.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled52.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled53.py --wdir
w2i
idx
idxs
indices
%runfile C:/Users/srivi/.spyder-py3/untitled53.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled54.py --wdir
input_data.shape
train_data.shape
len(train_data)
len(train_target)
len(predictions)
len(batch_target)
batch_input = torch.tensor(input_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.long)
batch_target = torch.tensor(target_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.float)

optimizer.zero_grad()
input_data = train_data
batch_size=500
iter_loss = 0
n_batches = len(input_data) // batch_size
n_batches
b_i = 1
batch_input = torch.tensor(input_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.long)
batch_target = torch.tensor(target_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.float)
target_data = train_target
batch_input = torch.tensor(input_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.long)

batch_target = torch.tensor(target_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.float)
len(batch_target)
len(batch_input)
optimizer.zero_grad()
            predictions = model(batch_input).squeeze(1)  # Ensure correct shape
            loss = criterion(predictions, batch_target)
len(predictions)
predictions = model(batch_input)
len(predictions)
%runfile C:/Users/srivi/.spyder-py3/untitled54.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled55.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled56.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled57.py --wdir
index
%runfile C:/Users/srivi/.spyder-py3/untitled57.py --wdir
seq_indices
%runfile C:/Users/srivi/.spyder-py3/untitled57.py --wdir
rnn
%runfile C:/Users/srivi/.spyder-py3/untitled57.py --wdir
(np.random.randn(hidden_size, input_size)).shape
input_size = 10 #dimension of the embedding layer to 10 
hidden_size = 20 
output_size = 1
seq_length = 1000
vocab_size = 15
(np.random.randn(hidden_size, input_size)).shape
(np.random.randn(hidden_size, hidden_size)).shape
input.shaüe
input.shape
%runfile C:/Users/srivi/.spyder-py3/untitled59.py --wdir
softmax(y_pred).shape, np.eye(15)[targets].T.shape
%runfile C:/Users/srivi/.spyder-py3/untitled57.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled59.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled60.py --wdir
self.Wxh, inputs[t]
%runfile C:/Users/srivi/.spyder-py3/untitled60.py --wdir
inputs
np.dot(self.Wxh:t, inputs[t])
np.dot(self.Wxh.T, inputs[t])
%runfile C:/Users/srivi/.spyder-py3/untitled60.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled59.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled57.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled56.py --wdir
%runfile c:/users/srivi/.spyder-py3/untitled61.py --wdir
self.embedding_dim
embedding_dim=128
np.random.randint(0, 26, size=50)
inputs = np.random.randint(0, 26, size=50)
vocab_size=26
embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
embedding.shape
inputs
idx=9
embedding[idx].reshape(-1, 1)
embedding[idx].shape
embedding[idx].reshape(-1, 1).shape
%runfile c:/users/srivi/.spyder-py3/untitled62.py --wdir
y_t
y_t.shape
x_t.shape
%runfile c:/users/srivi/.spyder-py3/untitled61.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled64.py --wdir
X_data
Y_data
data
X_tensor
X_tensor.shape
X_data.shape
len(X_data)
data.shape
len(data)
%runfile C:/Users/srivi/.spyder-py3/untitled64.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled65.py --wdir
pip install datasets
%runfile C:/Users/srivi/.spyder-py3/untitled65.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled70.py --wdir
df.shape
%runfile C:/Users/srivi/.spyder-py3/untitled70.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled65.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled70.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled72.py --wdir
loss, accuracy = model.evaluate(X_test, y_test)
%runfile C:/Users/srivi/.spyder-py3/untitled73.py --wdir
pd. set_option('display. max_columns', None)
pd.set_option('display. max_columns', None)
import pandas as pd
pd.set_option('display. max_columns', None)
pd.set_option('display.max_columns', None)
print("Matrix:", matrix)  # Tokenized matrix
matrix
pd.set_option('display.max_columns')
%runfile C:/Users/srivi/.spyder-py3/untitled73.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled72.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled77.py --wdir
%runfile C:/Users/srivi/.spyder-py3/untitled78.py --wdir