#DESCRIPTION (to be modified):
# Load old trained weights.
# For each layer, identify weights with smallest values ( w < T) and prune them, then retrain with L2 regularization and repeat


from utils import *
#^ imports plot, np and stuff

import tensorflow as tf
from tensorflow import Session, zeros, float32, reshape, shape
from tensorflow.examples.tutorials.mnist import input_data

def imshow(im):
    matplotlib.pyplot.imshow(im);
    matplotlib.pyplot.show();




#exec(open("vcPrune2.4.py").read())

#-------------------------------------------------------------------------#
#--------------------------Function Declarations--------------------------#

computerName='HEX'

def evin(i,plotit=True):
    print('Accuracy:');
    print(accuracy.eval(session=session,feed_dict={
            X: X_test[i:i+1], y:y_test[i:i+1]}));

    print('y_out:');
    print(session.run(y_out,feed_dict={X:X_test[i:i+1]}));
    print('y_softmax:');
    print(session.run(tf.nn.softmax(y_out),feed_dict={X:X_test[i:i+1]}));
    
    print('Actual: ');
    print(y_test[i:i+1])
    if plotit:
        imshow(X_test[i]);


def ev_mistakes():
    for i in range(0,655):
        if accuracy.eval(session=session,feed_dict={
            X: X_test[i:i+1], y:y_test[i:i+1]}) <1.0:

            print('Accuracy:');
            print(accuracy.eval(session=session,feed_dict={
                    X: X_test[i:i+1], y:y_test[i:i+1]}));

            print('y_out:');
            print(session.run(y_out,feed_dict={X:X_test[i:i+1]}));
            print('y_softmax:');
            print(session.run(tf.nn.softmax(y_out),feed_dict={X:X_test[i:i+1]}));
    
            print('Actual: ');
            print(y_test[i:i+1])
           
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


#exec(open("vcPrune2.4.py").read())
#____________________________________
#------------- main() ------------- #
#____________________________________



X_train, y_train, X_test, y_test = getCarData();



print('Constructing network\n');
#tf.device('/cpu:0');
train_N=np.shape(X_train)[0];
minibatch_size=128;
batches=int(np.ceil(train_N/minibatch_size));
train_prob=0.95;
in_w=96;
in_h=96;
out_dim=4;

X= tf.placeholder(tf.float32, shape=[None,in_w,in_h,3]);
y= tf.placeholder(tf.float32, shape=[None, out_dim]);

##input_conv1
W_conv1 = tf.Variable(np.load('weights/acc_977_972/W_conv1.npy'));
b_conv1 = tf.Variable(np.load('weights/acc_977_972/b_conv1.npy'));
W_conv2  =tf.Variable(np.load('weights/acc_977_972/W_conv2.npy'));
b_conv2 = tf.Variable(np.load('weights/acc_977_972/b_conv2.npy'));
W_d1 = tf.Variable(np.load('weights/acc_977_972/W_d1.npy'));
b_d1 = tf.Variable(np.load('weights/acc_977_972/b_d1.npy'));
W_d2 = tf.Variable(np.load('weights/acc_977_972/W_d2.npy'));
b_d2 = tf.Variable(np.load('weights/acc_977_972/b_d2.npy'));
W_out = tf.Variable(np.load('weights/acc_977_972/W_out.npy'));
b_out = tf.Variable(np.load('weights/acc_977_972/b_out.npy'));


dptprob = tf.placeholder(tf.float32)


h_conv1 = tf.nn.relu(conv2d(X, W_conv1)); 
h_pool1 = max_pool_2x2(h_conv1);


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)); 
h_pool2 = max_pool_2x2(h_conv2);


h_pool2_flat = tf.nn.dropout(tf.reshape(h_pool2, [-1, 24*24*32]),dptprob);
h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_d1));

h_d2 =  tf.nn.dropout(tf.nn.relu(tf.matmul(h_d1, W_d2)),dptprob);

y_out = tf.matmul(h_d2, W_out); # y_softmax handles scaling, so no activation here should be okay


lmb=0.35
nW= tf.cast(train_N ,tf.float32);
##loss
y_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out);
#ce= tf.reduce_mean(y_softmax) + (lmb/(2.0*nW))*(tf.norm(W_conv1)+tf.norm(W_conv2)+tf.norm(W_d1)+tf.norm(W_d2)+tf.norm(W_out));
ce= tf.reduce_mean(y_softmax) + (lmb/(2.0*nW))*(tf.square(tf.norm(W_conv1))+tf.square(tf.norm(W_conv2))+tf.square(tf.norm(W_d1))+tf.square(tf.norm(W_d2))+tf.square(tf.norm(W_out)));
#train_step= tf.train.GradientDescentOptimizer(0.005).minimize(ce);
train_step = tf.train.RMSPropOptimizer(0.001).minimize(ce);

#evaluation
y_true=tf.equal(tf.argmax(y_out,1), tf.argmax(y,1));
accuracy = tf.reduce_mean(tf.cast(y_true, tf.float64));

  
session=Session();
session.run(tf.global_variables_initializer()); 

train_epochs=60;
tune_epochs=40
tune_stages=10


best_val_accuracy=0.0;
best_test_accuracy=0.0;
save_weights=False;
epochs=train_epochs;
train_N=np.shape(X_train)[0];
minibatch_size=128; 
batches=int(np.ceil(train_N/minibatch_size));


print('Initial accuracy: ')

mean_test_acc=[];
mean_val_acc=[];

for i in range(0,327):    
    mean_val_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

for i in range(327,np.shape(y_test)[0]):    
    mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

ret_acc=np.zeros([tune_stages,train_epochs]);
tune_acc=np.zeros([tune_stages,tune_epochs]);
train_acc=np.zeros([tune_stages,train_epochs]);
tune_train_acc=np.zeros([tune_stages,tune_epochs]);
sprs=np.zeros([tune_stages]);

for stg in range(0,tune_stages):

    print('Stage %i/%i' %(stg+1,tune_stages))


    best_val_accuracy=0.0;
    best_test_accuracy=0.0;
    if stg>0:
        h_W_conv1=session.run(W_conv1);
        h_W_conv2=session.run(W_conv2);
        h_W_d1=session.run(W_d1);
        h_W_d2=session.run(W_d2);
        h_W_out=session.run(W_out);
        W_conv1 = tf.Variable(h_W_conv1);
        W_conv2 =  tf.Variable(h_W_conv2);
        W_d1 =  tf.Variable(h_W_d1);
        W_d2 =  tf.Variable(h_W_d2);
        W_out =  tf.Variable(h_W_out);

        h_conv1 = tf.nn.relu(conv2d(X, W_conv1)); 
        h_pool1 = max_pool_2x2(h_conv1);


        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)); 
        h_pool2 = max_pool_2x2(h_conv2);


        h_pool2_flat = tf.nn.dropout(tf.reshape(h_pool2, [-1, 24*24*32]),dptprob);
        h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_d1));

        h_d2 =  tf.nn.dropout(tf.nn.relu(tf.matmul(h_d1, W_d2)),dptprob);

        y_out = tf.matmul(h_d2, W_out); # y_softmax handles scaling, so no activation here should be okay



        ##loss
        y_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out);
        #ce= tf.reduce_mean(y_softmax) + (lmb/(2.0*nW))*(tf.norm(W_conv1)+tf.norm(W_conv2)+tf.norm(W_d1)+tf.norm(W_d2)+tf.norm(W_out));
        ce= tf.reduce_mean(y_softmax) + (lmb/(2.0*nW))*(tf.square(tf.norm(W_conv1))+tf.square(tf.norm(W_conv2))+tf.square(tf.norm(W_d1))+tf.square(tf.norm(W_d2))+tf.square(tf.norm(W_out)));
        #train_step= tf.train.GradientDescentOptimizer(0.005).minimize(ce);
        train_step = tf.train.RMSPropOptimizer(0.001).minimize(ce);

        #evaluation
        y_true=tf.equal(tf.argmax(y_out,1), tf.argmax(y,1));
        accuracy = tf.reduce_mean(tf.cast(y_true, tf.float64));

        session.close();
        session=Session();
        session.run(tf.global_variables_initializer()); 




        epochs=train_epochs

        for e in range(0, epochs):



          
            print("Epoch %i/%i" % (e+1,epochs));

            mean_loss=[];
            for batch in range(0, batches):
            
        
                train_accuracy = accuracy.eval(session=session,feed_dict={
                                X: X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)], 
                                y: y_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)],dptprob:1.0});
            
                mean_loss.append(train_accuracy);
            
           
                if batch%12==0:    
                    print('%d/%d | batch accuracy %g | mean epoch accuracy %g' % ((batch+1)*minibatch_size, train_N,
                                                                            train_accuracy, np.mean(mean_loss)));
                

                
                train_step.run(session=session,feed_dict={
                                    X: X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)], 
                                    y: y_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)],dptprob:train_prob})
        



            print('train accuracy: ')

            mean_test_acc=[];
            mean_val_acc=[];

            for i in range(0,327):    
                mean_val_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

            print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

            for i in range(327,np.shape(y_test)[0]):    
                mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

            print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

   

            if (best_val_accuracy+best_test_accuracy) <=(np.mean(mean_val_acc)+np.mean(mean_test_acc)):
                best_val_accuracy=np.mean(mean_val_acc);
                best_test_accuracy=np.mean(mean_test_acc);


            
            print('------ Best V/T accuracy: %g / %g ------' % (best_val_accuracy, best_test_accuracy));

            ret_acc[stg,e]=np.mean(mean_val_acc);
            train_acc[stg,e]= np.mean(mean_loss);





    #------ save gpu buffer data on host side
    h_W_conv1=session.run(W_conv1);
    h_W_conv2=session.run(W_conv2);
    h_W_d1=session.run(W_d1);
    h_W_d2=session.run(W_d2);
    h_W_out=session.run(W_out);

    total_removed=0.0
    total_params=np.size(h_W_d1);

    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)));     
    T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/2;     
   # T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/3;     
    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/4;     
    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/5;     
    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/6;     
    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/7;     
    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/8;     
    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/9;     
    #T1=(session.run(tf.reduce_max(W_d1) + tf.reduce_min(W_d1)))/10;     
 
    bmask=tf.greater(W_d1,T1);
    fbmask=tf.constant(session.run(tf.cast(tf.greater(W_d1,T1),tf.float32)));

    mask_W_d1=tf.cast( tf.greater(W_d1,T1),tf.float32);


    removed=session.run(tf.size(W_d1))-session.run(tf.reduce_sum(tf.cast(bmask,tf.float32)));
    print('W_d1: Removed ' + str(removed) + ' (' + str(np.round((removed/total_params)*100,4)) + '%) connections.');
    total_removed=total_removed+removed;




    print('Total parameter reduction: ' + str(total_removed) +' (' + str(np.round(total_removed/total_params,4)*100.0) + '%, ' + str(np.round(total_params/(total_params-total_removed),0)) + 'X) removed')
    sprs[stg]=np.round(total_removed/total_params,4)*100.0;

    #------ re-'wire' graph and initialize weights using previously saved buffers
    #------ add a multiplication with the mask to zero out w<T



    W_conv1 = tf.Variable(h_W_conv1);
    W_conv2 =  tf.Variable(h_W_conv2);
    W_d1 =  tf.Variable(h_W_d1);
    W_d2 =  tf.Variable(h_W_d2);
    W_out =  tf.Variable(h_W_out);


    h_conv1 = tf.nn.relu(conv2d(X, W_conv1)); 
    h_pool1 = max_pool_2x2(h_conv1);


    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)); 
    h_pool2 = max_pool_2x2(h_conv2);




    h_pool2_flat = tf.nn.dropout(tf.reshape(h_pool2, [-1, 24*24*32]),dptprob);
    h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, tf.multiply(W_d1,fbmask)));

    h_d2 =  tf.nn.dropout(tf.nn.relu(tf.matmul(h_d1, W_d2)),dptprob);


    y_out = tf.matmul(h_d2, W_out); # y_softmax handles scaling, so no activation here should be okay


    ##loss
    y_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out);
    #ce= tf.reduce_mean(y_softmax) 
    #ce= tf.reduce_mean(y_softmax) + (lmb/(2.0*nW))*(tf.norm(W_conv1)+tf.norm(W_conv2)+tf.norm(W_d1)+tf.norm(W_d2)+tf.norm(W_out));
    ce= tf.reduce_mean(y_softmax) + (lmb/(2.0*nW))*(tf.square(tf.norm(W_conv1))+tf.square(tf.norm(W_conv2))+tf.square(tf.norm(W_d1))+tf.square(tf.norm(W_d2))+tf.square(tf.norm(W_out)));
    #train_step= tf.train.GradientDescentOptimizer(0.005).minimize(ce);
    train_step = tf.train.RMSPropOptimizer(0.001).minimize(ce);

    #evaluation
    y_true=tf.equal(tf.argmax(y_out,1), tf.argmax(y,1));
    accuracy = tf.reduce_mean(tf.cast(y_true, tf.float64));

    session.close() ;
    session=Session();
    session.run(tf.global_variables_initializer()); 





    print('Pruned accuracy: ')

    mean_test_acc=[];
    mean_val_acc=[];


    for i in range(0,327):    
        mean_val_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

    print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

    for i in range(327,np.shape(y_test)[0]):    
        mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

    print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

    
    #tune connections    

    save_weights=False;

    epochs=tune_epochs

    for e in range(0, epochs):



          
        print("Epoch %i/%i" % (e+1,epochs));

        mean_loss=[];
        for batch in range(0, batches):
            
        
            train_accuracy = accuracy.eval(session=session,feed_dict={
                            X: X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)], 
                            y: y_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)],dptprob:1.0});
            
            mean_loss.append(train_accuracy);
            
           
            if batch%12==0:    
                print('%d/%d | batch accuracy %g | mean epoch accuracy %g' % ((batch+1)*minibatch_size, train_N,
                                                                        train_accuracy, np.mean(mean_loss)));
                

                
            train_step.run(session=session,feed_dict={
                                X: X_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)], 
                                y: y_train[batch*minibatch_size:min((batch+1)*minibatch_size,train_N)],dptprob:1.0})
        



        print('Tuned accuracy: ')

        mean_test_acc=[];
        mean_val_acc=[];

        for i in range(0,327):    
            mean_val_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

        print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

        for i in range(327,np.shape(y_test)[0]):    
            mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1],dptprob:1.0}));

        print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

   

        if (best_val_accuracy+best_test_accuracy) <=(np.mean(mean_val_acc)+np.mean(mean_test_acc)):
            best_val_accuracy=np.mean(mean_val_acc);
            best_test_accuracy=np.mean(mean_test_acc);

        print('------ Best V/T accuracy: %g / %g ------' % (best_val_accuracy, best_test_accuracy));


        tune_acc[stg,e]=np.mean(mean_val_acc);
        tune_train_acc[stg,e]= np.mean(mean_loss);







#---- final verification to make sure W_d1 really is sparse. W_d1 is initialized with the sparse matrix. W_d1 output is printed at the end and the mask alongside it to verify

print("--Final verification--")
h_W_conv1=session.run(W_conv1);
h_W_conv2=session.run(W_conv2);
h_W_d1=session.run(tf.multiply(W_d1,fbmask));
h_W_d2=session.run(W_d2);
h_W_out=session.run(W_out);

W_conv1 = tf.Variable(h_W_conv1);
W_conv2 =  tf.Variable(h_W_conv2);
W_d1 = tf.Variable(h_W_d1);
W_d2 =  tf.Variable(h_W_d2);
W_out =  tf.Variable(h_W_out);


h_conv1 = tf.nn.relu(conv2d(X, W_conv1)); 
h_pool1 = max_pool_2x2(h_conv1);


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)); 
h_pool2 = max_pool_2x2(h_conv2);


h_pool2_flat = tf.reshape(h_pool2, [-1, 24*24*32]);
h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_d1));

h_d2 = tf.nn.relu(tf.matmul(h_d1, W_d2));

y_out = tf.matmul(h_d2, W_out); # y_softmax handles scaling, so no activation here should be okay


##loss
y_softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out);
ce= tf.reduce_mean(y_softmax);
#train_step= tf.train.GradientDescentOptimizer(0.005).minimize(ce);
train_step = tf.train.RMSPropOptimizer(0.001).minimize(ce);

#evaluation
y_true=tf.equal(tf.argmax(y_out,1), tf.argmax(y,1));
accuracy = tf.reduce_mean(tf.cast(y_true, tf.float64));

  
session.close() ;
session=Session();
session.run(tf.global_variables_initializer()); 

for i in range(0,327):    
    mean_val_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1]}));

print('------ Mean validation accuracy(%i): %g ------' % (i,np.mean(mean_val_acc)));

for i in range(327,np.shape(y_test)[0]):    
    mean_test_acc.append(accuracy.eval(session=session,feed_dict={X: X_test[i:i+1], y: y_test[i:i+1]}));

print('------ Mean test accuracy(%i): %g ------' % (i,np.mean(mean_test_acc)));

print((np.mean(mean_test_acc) + np.mean(mean_val_acc))/2.0)
print('Total parameter reduction: ' + str(total_removed) +' (' + str(np.round(total_removed/total_params,4)*100.0) + '%, ' + str(np.round(total_params/(total_params-total_removed),0)) + 'X) removed')

#print(session.run(fbmask))
#print(session.run(W_d1))


#run this, get accuracy/loss at each stage and make graph

imsave(np.reshape(ret_acc,[tune_stages,train_epochs],'C'),'pruneacc1/retacc.csv')
imsave(np.reshape(train_acc,[tune_stages,train_epochs],'C'),'pruneacc1/rettrainacc.csv')
imsave(np.reshape(tune_acc,[tune_stages,tune_epochs],'C'),'pruneacc1/tuneacc.csv')
imsave(np.reshape(tune_train_acc,[tune_stages,tune_epochs],'C'),'pruneacc1/tunetrainacc.csv')
imsave(np.reshape(sprs,[tune_stages],'C'),'pruneacc1/sprs.csv')

