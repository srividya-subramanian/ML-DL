%runfile C:/Users/srivi/.spyder-py3/untitled33.py --wdir
w =pd.DataFrame([(
      [0.9,-0.1, 0.1],
      [-0.2, 0.8, 0.4]
      )])
w
np.array([[1, 2], [3, 4]])
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
w =np.array([[0.9,-0.1, 0.1],[-0.2, 0.8, 0.4]])
w
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
wsum
wsum.shape
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
wsum
len(wsum)
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
y_pred
zero
wsum
zero.tolist
z =zero.tolist
z
z =zero.tolist()
z
max(zero, wsum)
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
y_üred
y_pred
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
y_pred
%runfile C:/Users/srivi/.spyder-py3/untitled31.py --wdir
pip install chatterbot
%runfile C:/Users/srivi/.spyder-py3/untitled36.py --wdir

## ---(Sun Feb  2 17:27:54 2025)---
%runfile C:/Users/srivi/.spyder-py3/chatbot.py --wdir
pip install chatterbot
python
pip show python
pip install tkinter
%runfile C:/Users/srivi/.spyder-py3/chatbot.py --wdir
import nltk
from nltk.chat.util import Chat, reflections
%runfile C:/Users/srivi/.spyder-py3/chatbot.py --wdir
import tensorflow keras pickle nltk
import tensorflow
import keras
import nltk, pickle
%runfile C:/Users/srivi/.spyder-py3/untitled0.py --wdir
%runfile C:/Users/srivi/.spyder-py3/chatbot_NN.py --wdir
nltk.download('punkt_tab')
%runfile C:/Users/srivi/.spyder-py3/chatbot_NN.py --wdir
intents
%runfile C:/Users/srivi/.spyder-py3/chatbot_NN.py --wdir

## ---(Mon Feb  3 11:16:35 2025)---
%runfile C:/Users/srivi/.spyder-py3/untitled0.py --wdir
y_pred = forward_pass(X,w)
%runfile C:/Users/srivi/.spyder-py3/untitled1.py --wdir
y_pred = forward_pass(X,w)
X.shape
w,shape
w,´.shape
w.shape
%runfile C:/Users/srivi/.spyder-py3/untitled1.py --wdir
y_pred.shape
y
y.shape
n = len(y)
n
err = calculate_error(y, y_pred)
err. shape
y.shape
x.shape
X.shape
w. shape
w
err
x * err
X * err
X.shape
err.shape
np.dot(x,err)
np.dot(X,err)
%runfile C:/Users/srivi/.spyder-py3/untitled1.py --wdir
np.dot(xX,err) * lr
np.dot(X,err) * lr
w
x
X.shape, w.shape, np.dot(x,err).shape
X.shape, w.shape, np.dot(X,err).shape
%runfile C:/Users/srivi/.spyder-py3/untitled1.py --wdir
w
w_new
%runfile C:/Users/srivi/.spyder-py3/untitled1.py --wdir
w_new
%runfile C:/Users/srivi/.spyder-py3/untitled1.py --wdir
error
error.shape
error.shape()
%runfile C:/Users/srivi/.spyder-py3/untitled1.py --wdir
w
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
train.shape
test.shape
test.head(5)
test['label']
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
w1
max(w1)
np.max(w1)
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
X_train.shape
w.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
2w
w
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
w.shape
X_train.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
y_train.shape
y_pred.shape
y_pred = forward_pass(X_train,w)
y_pred.shape
y_pred
np.dot(X_train,w)
train.drop('label', axis=1).T
(train.drop('label', axis=1).T).shape
w.shape
np.dot((train.drop('label', axis=1).T),w)
w.dot(train.drop('label', axis=1).T)
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
y_pred
y_pred.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
y_pred.shape
y_train.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
y_pred.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
y.shape
y_pred.shape
y_train.shape
%runfile C:/Users/srivi/.spyder-py3/untitled5.py --wdir
weights.shape
images.shape
softmax(np.dot(images, weights))
softmax(np.dot(images, weights)).shape
predictions.shape
predictions = predict(images, weights)
predictions.shape
labels.shape
train_labels[:1000]
train_labels[:1000].shape
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
w1.shape
w2.shape
a1.shape
a1
%runfile C:/Users/srivi/.spyder-py3/untitled6.py --wdir
y_pred.shape
x_train.shape
X_train.shape
z1.shape, z2.shape
Y=y_train
X= x_train
X= X_train
w1
gradient_descent(X, Y, iterations, alpha)
z1 = w1.dot(X) + b1
a1 = ReLU(z1)
z2 = w2.dot(a1) + b2
a2 = softmax(z2)
one_hot_Y = np.zeros((Y.size, Y.max() + 1))
one_hot_Y.shape
one_hot_Y.head(5)
one_hot_Y[np.arange(Y.size), Y] = 1
one_hot_Y.shape
one_hot_Y = one_hot_Y.T
one_hot_Y.shape
m,n
a1
z1
a2
z2
labels = np.eye(10)[train_labels[:1000]]
labels.shape
predictions.shape
%runfile C:/Users/srivi/.spyder-py3/untitled2.py --wdir
np.dot(x,err)
np.dot(X_train,err)
err
err.shape
X_train.shape
error.shape
%runfile C:/Users/srivi/.spyder-py3/untitled5.py --wdir
error.shape
predictions.shape
error = predictions - labels
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