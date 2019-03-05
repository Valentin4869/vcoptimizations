
from utils import *
#^ imports plot, np and stuff

import tensorflow as tf
from tensorflow import Session, zeros, float32, reshape, shape
from tensorflow.examples.tutorials.mnist import input_data

def imshow(im):
    matplotlib.pyplot.imshow(im);
    matplotlib.pyplot.show();

#load original weights
# define two nets: original architecture and sep architecture
# 'train' for separable kernels and compare accuracy
# loss is huge, but accuracy seems to recover early on even with large loss values.



#exec(open("vcSepKernels.py").read())

#-------------------------------------------------------------------------#
#--------------------------Function Declarations--------------------------#

computerName='HEX'

def evin(i,plotit=True):
    print('Accuracy:');
    print(accuracy.eval(session=session,feed_dict={
            X: X_test[i:i+1], y:y_test[i:i+1]}));
    print(session.run(y_out,feed_dict={X:X_test[i:i+1]}));
    print('Actual: ');
    print(y_test[i:i+1])
    if plotit:
        imshow(X_test[i]);

def weight_variable(shape):
#truncated normal error
  if computerName=='HEX': #home computer, recent tf version
    initial = tf.truncated_normal(shape, stddev=0.1)
    
  elif computerName=='mir-OptiPlex-9020': #school computer, old tf version   
    initial = tf.random_normal(shape, stddev=0.1)
    
  
  return tf.Variable(initial)

def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):

  return tf.nn.conv2d(x,
                      W,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

def max_pool_2x2(x):

  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')


#-------------------------------------------------------------------------#


#____________________________________
#------------- main() ------------- #
#____________________________________

#today before leaving: load weights, define loss for each conv layer, find approximations, test accuracy with sep conv, batched

X_train, y_train, X_test, y_test = getCarData();


print('Constructing network\n');
#tf.device('/cpu:0');
epochs=1000;
gridsearch=np.zeros([1,16]);

glosses1=np.zeros([16,epochs])
glosses2=np.zeros([16,epochs])
gacc=np.zeros([16,epochs])


train_N=np.shape(X_train)[0];
minibatch_size=128;
batches=int(np.ceil(train_N/minibatch_size));

#next day: run pruning, record accuracy at each sparsity level for plot(sparsity/stage/accuracy)
for K in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:

    d=5;

    C=3;
    N=32;
    imh=96;
    imw=96;
    in_dim=imw*imh;
    out_dim=4;
    #conv
    conv1_fw= 5;
    conv2_fw= 5;



    #X= tf.placeholder(tf.float32, shape=[None, in_dim]);
    X= tf.placeholder(tf.float32, shape=[None,96,96,3]);
    y= tf.placeholder(tf.float32, shape=[None, out_dim]);



    ##input_conv1

    W_conv1 = tf.Variable(np.load('weights/acc_977_972/W_conv1.npy'));
    W_conv2  =tf.Variable(np.load('weights/acc_977_972/W_conv2.npy'));
    b_conv1  =tf.Variable(np.load('weights/acc_977_972/b_conv1.npy'));
    b_conv2  =tf.Variable(np.load('weights/acc_977_972/b_conv2.npy'));
    W_d1 = tf.Variable(np.load('weights/acc_977_972/W_d1.npy'));
    b_d1 = tf.Variable(np.load('weights/acc_977_972/b_d1.npy'));
    W_d2 = tf.Variable(np.load('weights/acc_977_972/W_d2.npy'));
    b_d2 = tf.Variable(np.load('weights/acc_977_972/b_d2.npy'));
    W_out = tf.Variable(np.load('weights/acc_977_972/W_out.npy'));
    b_out = tf.Variable(np.load('weights/acc_977_972/b_out.npy'));

    h_conv1 = tf.nn.relu(conv2d(X, W_conv1)+b_conv1); #h_conv1 = tf.nn.relu(conv2d(X, W_conv1,) + b_conv1);
    h_pool1 = max_pool_2x2(h_conv1);

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2);
    h_pool2 = max_pool_2x2(h_conv2);
    h_pool2_flat = tf.reshape(h_pool2, [-1, 24*24*32]);
    h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_d1) + b_d1);


    keep_prob_c2d = tf.placeholder(tf.float32)
    h_d1_dpt = tf.nn.dropout(h_d1, keep_prob_c2d)

    h_d2 = tf.nn.relu(tf.matmul(h_d1_dpt, W_d2) + b_d2);

    keep_prob_d1d2 = tf.placeholder(tf.float32)
    h_d2_dpt = tf.nn.dropout(h_d2, keep_prob_d1d2)
    y_out = tf.matmul(h_d2_dpt, W_out) + b_out; #this one works



    v1=tf.Variable(tf.truncated_normal([d,1,C,K],stddev=0.001));
    h1=tf.Variable(tf.truncated_normal([1,d,K,N],stddev=0.001));
    v2=tf.Variable(tf.truncated_normal([d,1,N,K],stddev=0.001));
    h2=tf.Variable(tf.truncated_normal([1,d,K,N],stddev=0.001));
    spW_d1 = tf.Variable(np.load('weights/acc_977_972/W_d1.npy'));
    spb_d1 = tf.Variable(np.load('weights/acc_977_972/b_d1.npy'));
    spW_d2 = tf.Variable(np.load('weights/acc_977_972/W_d2.npy'));
    spb_d2 = tf.Variable(np.load('weights/acc_977_972/b_d2.npy'));
    spW_out = tf.Variable(np.load('weights/acc_977_972/W_out.npy'));
    spb_out = tf.Variable(np.load('weights/acc_977_972/b_out.npy'));

    sph_conv1=tf.nn.relu(conv2d(conv2d(X,v1),h1));
    sph_pool1=max_pool_2x2(sph_conv1);

    sph_conv2=tf.nn.relu(conv2d(conv2d(sph_pool1,v2),h2));
    sph_pool2=max_pool_2x2(sph_conv2);
    sph_pool2_flat = tf.reshape(sph_pool2, [-1, 24*24*32]);
    sph_d1 = tf.nn.relu(tf.matmul(sph_pool2_flat, spW_d1) + spb_d1);


    spkeep_prob_c2d = tf.placeholder(tf.float32)
    sph_d1_dpt = tf.nn.dropout(sph_d1, spkeep_prob_c2d)

    sph_d2 = tf.nn.relu(tf.matmul(sph_d1_dpt, spW_d2) + spb_d2);

    spkeep_prob_d1d2 = tf.placeholder(tf.float32)
    sph_d2_dpt = tf.nn.dropout(sph_d2, spkeep_prob_d1d2)
    spy_out = tf.matmul(sph_d2_dpt, spW_out) + spb_out; #this one works



    ##loss
    y_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out);
    ce= tf.reduce_mean(y_softmax);
    train_step = tf.train.RMSPropOptimizer(0.001,0.992).minimize(ce);

    y_true=tf.equal(tf.argmax(y_out,1), tf.argmax(y,1));
    accuracy = tf.reduce_mean(tf.cast(y_true, tf.float64));
  


    spy_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=spy_out);
    spce= tf.reduce_mean(spy_softmax);
    spy_true=tf.equal(tf.argmax(spy_out,1), tf.argmax(y,1));
    spaccuracy = tf.reduce_mean(tf.cast(spy_true, tf.float64));
  

    rndIndx=np.random.permutation(np.shape(X_train)[0]);

    #gloss=tf.reduce_sum(tf.reduce_sum(tf.norm(h_pool1- sph_pool1, axis=[1,2]),axis=0));
    #gloss2=tf.reduce_sum(tf.reduce_sum(tf.norm(h_pool2- sph_pool2, axis=[1,2]),axis=0));

    gloss=tf.reduce_mean((tf.norm(h_pool1- sph_pool1)));
    gloss2=tf.reduce_mean((tf.norm(h_pool2- sph_pool2)));
    

    #optsteph= tf.train.RMSPropOptimizer(0.001).minimize(gloss,var_list=[h1]);
    #optstepv= tf.train.RMSPropOptimizer(0.001).minimize(gloss,var_list=[v1]);
    #optsteph2= tf.train.RMSPropOptimizer(0.001).minimize(gloss2,var_list=[h2]);
    #optstepv2= tf.train.RMSPropOptimizer(0.001).minimize(gloss2,var_list=[v2]);

    #epoch 10, 75
    #optsteph= tf.train.AdagradOptimizer(0.1).minimize(gloss,var_list=[h1]);
    #optstepv= tf.train.AdagradOptimizer(0.1).minimize(gloss,var_list=[v1]);
    #optsteph2= tf.train.AdagradOptimizer(0.1).minimize(gloss2,var_list=[h2]);
    #optstepv2= tf.train.AdagradOptimizer(0.1).minimize(gloss2,var_list=[v2]);

    #optsteph= tf.train.AdadeltaOptimizer(1.0).minimize(gloss,var_list=[h1]);
    #optstepv= tf.train.AdadeltaOptimizer(1.0).minimize(gloss,var_list=[v1]);
    #optsteph2= tf.train.AdadeltaOptimizer(1.0).minimize(gloss2,var_list=[h2]);
    #optstepv2= tf.train.AdadeltaOptimizer(1.0).minimize(gloss2,var_list=[v2]);

    optsteph= tf.train.RMSPropOptimizer(0.001,0.992).minimize(gloss,var_list=[h1]);
    optstepv= tf.train.RMSPropOptimizer(0.001,0.992).minimize(gloss,var_list=[v1]);
    optsteph2= tf.train.RMSPropOptimizer(0.001,0.992).minimize(gloss2,var_list=[h2]);
    optstepv2= tf.train.RMSPropOptimizer(0.001,0.992).minimize(gloss2,var_list=[v2]);


    sptrain_step1 = tf.train.RMSPropOptimizer(0.001,0.995).minimize(gloss,var_list=[h1,v1]);
    sptrain_step2 = tf.train.RMSPropOptimizer(0.001,0.995).minimize(gloss2,var_list=[h2,v2]);
    hWf=np.load('hW_99.npy');


    session=Session();
    session.run(tf.global_variables_initializer()); 

    mean_test_acc=[];
    mean_val_acc=[];
    print('Initial original...')

    for i in range(0,300):    
        mean_val_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));
    print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

    for i in range(300,np.shape(y_test)[0]):    
        mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],keep_prob_c2d:1.0,keep_prob_d1d2:1.0}));
    print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));


    print('Initial sep...')

    for i in range(0,300):    
        mean_val_acc.append(spaccuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],spkeep_prob_c2d:1.0,spkeep_prob_d1d2:1.0}));
    print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

    for i in range(300,np.shape(y_test)[0]):    
        mean_test_acc.append(spaccuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],spkeep_prob_c2d:1.0,spkeep_prob_d1d2:1.0}));
    print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));


 

  
    for e in range(0, epochs):
     
        print("\nEpoch %i/%i" % (e+1,epochs));
        epoch_loss1=[]
        epoch_loss2=[]



        for batch in range(0, batches):
            

            #optsteph.run(feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]},session=session);
            #optstepv.run(feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]},session=session);
            #optsteph2.run(feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]},session=session);
            #optstepv2.run(feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]},session=session);

            sptrain_step1.run(feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]},session=session);
            sptrain_step2.run(feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]},session=session);


            epoch_loss1.append(session.run(gloss,feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]}))    
            epoch_loss2.append(session.run(gloss2,feed_dict={X:X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)]}))   
                 
            if batch%12==0:    
                print('%d/%d | batch losses %g, %g ' % ((batch+1)*minibatch_size, train_N,
                                                                        epoch_loss1[-1],
                                                                        epoch_loss2[-1]));

                
        glosses1[K-1,e]=np.mean(epoch_loss1);
        glosses2[K-1,e]=np.mean(epoch_loss2);

        mean_test_acc=[];
        mean_val_acc=[];
        for i in range(0,300):    
            mean_val_acc.append(spaccuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],spkeep_prob_c2d:1.0,spkeep_prob_d1d2:1.0}));

        print('------ Mean sep validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

        for i in range(300,np.shape(y_test)[0]):    
            mean_test_acc.append(spaccuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],spkeep_prob_c2d:1.0,spkeep_prob_d1d2:1.0}));

        print('------ Mean sep test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

        print('------ Mean sep loss1: %g ------' % (glosses1[K-1,e]));
        print('------ Mean sep loss2: %g ------' % (glosses2[K-1,e]));

        gridsearch[0,K-1]=  (np.mean(mean_val_acc)+np.mean(mean_test_acc))/2.0;

        gacc[K-1,e]=gridsearch[0,K-1];

    session.close();        
    print(gridsearch);

print(gridsearch);

imsave(np.reshape(glosses1,[1,16*epochs],'F'),'sclosses3/glosses1.csv')
imsave(np.reshape(glosses2,[1,16*epochs],'F'),'sclosses3/glosses2.csv')
