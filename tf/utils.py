import os
import glob
import socket
import numpy as np
from numpy import reshape, shape
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import Session
import tensorflow as tf
import mat4py as mt

eps=np.finfo(np.float).eps

#np.random.seed(2); print('--WARNING: using numpy random seed --');



#------------------fix numpy----------------------
def dims(x):
    return shape(shape(x))[0];

def permute(x,axes):
    return np.moveaxis(x,list(range(np.size(np.shape(x)))),axes);


#-------------------------------------------------

def tsave(fname,tensor_x,session):
    host_x= session.run(tensor_x);
    np.save(fname,host_x);

#matlab style

def ffilter (x):
    return np.fliplr(np.flip(x,0));

def loadmat(fname,s):
    w=mt.loadmat(fname)['data'];
    sz=np.shape(s)[0];
    #use reverse shape then transpose to correct shape (s); 
    #This is the only way to get to correct order from experimentation
    data=np.reshape(w,np.fliplr([s])[0]);
    a=np.fliplr([range(0,sz)])[0];
    print(a);
    data=np.transpose(data,axes=a);

    return data

def loadmat2(fname,s):
    w=mt.loadmat(fname)['data'];
    sz=np.shape(s)[0];
    #use reverse shape then transpose to correct shape (s); 
    #This is the only way to get to correct order from experimentation
    data=np.reshape(w,s);
      

    return data

def max_pool_2x2(x):

  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

def run(fn):
    exec(open(fn+".py").read())

def imsave(res,fname='c.csv'):
  
    if dims(res)<3:
        np.savetxt(fname, res, delimiter=",") ;
    elif dims(res)==3:
        print('Warning: saving only first channel');
        np.savetxt(fname, res[:,:,0], delimiter=",");
    elif dims(res)==4:
        print('Warning: saving only first feature map (4D input)');
        np.savetxt(fname, res[0,:,:,0], delimiter=",");

    else:
        print('can\'t save this...');

def imshow(im):
    matplotlib.pyplot.imshow(np.squeeze(im));
    matplotlib.pyplot.show();

def imread(fname):
    im=plt.imread(fname)
    im = (im - 0.0) /255.0 ;#im2double
    return im;

def imread2(fname):
    im=plt.imread(fname)
    im = (im - 0.0) /255.0 ;#im2double
    return reshape(im,[1,shape(im)[0],shape(im)[1],shape(im)[2]]);

def getClassStr(class_i):
    if class_i==0:
        return 'BUS';
    elif class_i==1:
        return 'NORMAL';
    elif class_i==2:
        return 'TRUCK';
    elif class_i==3:
        return 'VAN';

def getClassStrCIFAR10(class_i):
    if class_i==0:
        return 'AIRPLANE';
    elif class_i==1:
        return 'AUTOMOBILE';
    elif class_i==2:
        return 'BIRD';
    elif class_i==3:
        return 'CAT';
    elif class_i==4:
        return 'DEER';
    elif class_i==5:
        return 'DOG';
    elif class_i==6:
        return 'FROG';
    elif class_i==7:
        return 'HORSE';
    elif class_i==8:
        return 'SHIP';
    elif class_i==9:
        return 'TRUCK';

def make_one_hot(labels):
    num_labels=np.max(labels) +1;
    num_items=np.max(np.shape(labels))
    y=np.zeros((num_items,num_labels));

    y[np.arange(0,num_items),np.array(labels)[np.argmin(np.shape(labels)),:]]=1.0;

    return y;

def tfSession(session):

    session=Session();
    session.run(tf.global_variables_initializer()); 



#used to be:
# 
#def getCarData(tgtid=False):
#    path_train='C:/0/vc/DATA/TRAIN/';
#    path_test='C:/0/vc/DATA/TEST/';        

def getCarData(path_train='C:/0/vc/DATA/TRAIN/', path_test='C:/0/vc/DATA/TEST/',tgtid=False,shuffle=True):
    #path_train='C:/0/vc/DATA/TRAIN/';
    #path_test='C:/0/vc/DATA/TEST/';        


    print('Collecting Data');
    train_fnamelist=os.listdir(path_train);
    test_fnamelist=os.listdir(path_test);

    BUS_fnamelist_train=glob.glob(path_train+'BUS/*.jpg');
    NORMAL_fnamelist_train=glob.glob(path_train+'NORMALCAR/*.jpg');
    TRUCK_fnamelist_train=glob.glob(path_train+'TRUCK/*.jpg');
    VAN_fnamelist_train=glob.glob(path_train+'VAN/*.jpg');

    BUS_fnamelist_test=glob.glob(path_test+'BUS/*.jpg');
    NORMAL_fnamelist_test=glob.glob(path_test+'NORMALCAR/*.jpg');
    TRUCK_fnamelist_test=glob.glob(path_test+'TRUCK/*.jpg');
    VAN_fnamelist_test=glob.glob(path_test+'VAN/*.jpg');

    X_train = [];
    y_train= np.zeros([len(BUS_fnamelist_train)+
    len(NORMAL_fnamelist_train)+len(TRUCK_fnamelist_train)+len(VAN_fnamelist_train),4])

    X_test=[];
    y_test= np.zeros([len(BUS_fnamelist_test)+
    len(NORMAL_fnamelist_test)+len(TRUCK_fnamelist_test)+len(VAN_fnamelist_test),4])
    
    offset=0;
  
 
    for i in range(offset,len(BUS_fnamelist_train)):
        X_train.append(plt.imread(BUS_fnamelist_train[i]));
        y_train[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_train)):
        X_train.append(plt.imread(NORMAL_fnamelist_train[i]));
        y_train[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_train)):
        X_train.append(plt.imread(TRUCK_fnamelist_train[i]));
        y_train[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_train)):
        X_train.append(plt.imread(VAN_fnamelist_train[i]));
        y_train[i+offset][3]=1;


    offset=0;
    
    for i in range(offset,len(BUS_fnamelist_test)):
        X_test.append(plt.imread(BUS_fnamelist_test[i]));
        y_test[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_test)):
        X_test.append(plt.imread(NORMAL_fnamelist_test[i]));
        y_test[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_test)):
        X_test.append(plt.imread(TRUCK_fnamelist_test[i]));
        y_test[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_test)):
        X_test.append(plt.imread(VAN_fnamelist_test[i]));
        y_test[i+offset][3]=1;


    print('Shuffling...')
    #Shuffle
    
    # np.random.permutation(N): 
    #   list of consecutive integers from 0 to N-1
    #   in a random permutation
  
    if shuffle:
        rndIdx=np.random.permutation((np.shape(X_train))[0]); 
        cX_train=np.copy(X_train);
        cy_train=np.copy(y_train);

        for i in range(0,np.shape(X_train)[0]):
            X_train[i]=cX_train[rndIdx[i]];
            y_train[i]=cy_train[rndIdx[i]];

        rndIdx=np.random.permutation((np.shape(X_test))[0]); 
        cX_test=np.copy(X_test);
        cy_test=np.copy(y_test);
   
    

   

        for i in range(0,np.shape(X_test)[0]):
            X_test[i]=cX_test[rndIdx[i]];
            y_test[i]=cy_test[rndIdx[i]];

            
    
    if tgtid:
        return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
    else:
          
        return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)


def getCarData2( path_train='C:/Users/Mir/Documents/MATLAB/DATA aug5/TRAIN/',   path_test='C:/Users/Mir/Documents/MATLAB/DATA aug5/TEST/', path_validation='C:/Users/Mir/Documents/MATLAB/DATA aug5/VALIDATION/',tgtid=False):
    #path_train='C:/0/vc/DATA/TRAIN/';
    #path_test='C:/0/vc/DATA/TEST/';        

    #path_train='C:/Users/Mir/Documents/MATLAB/DATA aug5/TRAIN/'
    #path_test='C:/Users/Mir/Documents/MATLAB/DATA aug5/TEST/'
    #path_validation='C:/Users/Mir/Documents/MATLAB/DATA aug5/VALIDATION/'

    print('Path: '+ path_train)
    print('Collecting Data');
    train_fnamelist=os.listdir(path_train);
    test_fnamelist=os.listdir(path_test);
    validation_fnamelist=os.listdir(path_validation);

    BUS_fnamelist_train=glob.glob(path_train+'BUS/*.jpg');
    NORMAL_fnamelist_train=glob.glob(path_train+'NORMALCAR/*.jpg');
    TRUCK_fnamelist_train=glob.glob(path_train+'TRUCK/*.jpg');
    VAN_fnamelist_train=glob.glob(path_train+'VAN/*.jpg');

    BUS_fnamelist_test=glob.glob(path_test+'BUS/*.jpg');
    NORMAL_fnamelist_test=glob.glob(path_test+'NORMALCAR/*.jpg');
    TRUCK_fnamelist_test=glob.glob(path_test+'TRUCK/*.jpg');
    VAN_fnamelist_test=glob.glob(path_test+'VAN/*.jpg');

    BUS_fnamelist_validation=glob.glob(path_validation+'BUS/*.jpg');
    NORMAL_fnamelist_validation=glob.glob(path_validation+'NORMALCAR/*.jpg');
    TRUCK_fnamelist_validation=glob.glob(path_validation+'TRUCK/*.jpg');
    VAN_fnamelist_validation=glob.glob(path_validation+'VAN/*.jpg');

    X_train = [];
    y_train= np.zeros([len(BUS_fnamelist_train)+
    len(NORMAL_fnamelist_train)+len(TRUCK_fnamelist_train)+len(VAN_fnamelist_train),4])

    X_test=[];
    y_test= np.zeros([len(BUS_fnamelist_test)+
    len(NORMAL_fnamelist_test)+len(TRUCK_fnamelist_test)+len(VAN_fnamelist_test),4])
    
    X_validation = [];
    y_validation= np.zeros([len(BUS_fnamelist_validation)+
    len(NORMAL_fnamelist_validation)+len(TRUCK_fnamelist_validation)+len(VAN_fnamelist_validation),4])

    offset=0;
  
 
    for i in range(offset,len(BUS_fnamelist_train)):
        X_train.append(plt.imread(BUS_fnamelist_train[i]));
        y_train[i+offset][0]=1; #offset=0; just for consistency

        
        
    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_train)):
        X_train.append(plt.imread(NORMAL_fnamelist_train[i]));
        y_train[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_train)):
        X_train.append(plt.imread(TRUCK_fnamelist_train[i]));
        y_train[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_train)):
        X_train.append(plt.imread(VAN_fnamelist_train[i]));
        y_train[i+offset][3]=1;


    offset=0;
    
    for i in range(offset,len(BUS_fnamelist_test)):
        X_test.append(plt.imread(BUS_fnamelist_test[i]));
        y_test[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_test)):
        X_test.append(plt.imread(NORMAL_fnamelist_test[i]));
        y_test[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_test)):
        X_test.append(plt.imread(TRUCK_fnamelist_test[i]));
        y_test[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_test)):
        X_test.append(plt.imread(VAN_fnamelist_test[i]));
        y_test[i+offset][3]=1;


    offset=0;
    
    for i in range(offset,len(BUS_fnamelist_validation)):
        X_validation.append(plt.imread(BUS_fnamelist_validation[i]));
        y_validation[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_validation)):
        X_validation.append(plt.imread(NORMAL_fnamelist_validation[i]));
        y_validation[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_validation)):
        X_validation.append(plt.imread(TRUCK_fnamelist_validation[i]));
        y_validation[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_validation)):
        X_validation.append(plt.imread(VAN_fnamelist_validation[i]));
        y_validation[i+offset][3]=1;

    print('Shuffling...')
    #Shuffle
    
    # np.random.permutation(N): 
    #   list of consecutive integers from 0 to N-1
    #   in a random permutation
  
    rndIdx=np.random.permutation((np.shape(X_train))[0]); 
    cX_train=np.copy(X_train);
    cy_train=np.copy(y_train);

    for i in range(0,np.shape(X_train)[0]):
        X_train[i]=cX_train[rndIdx[i]];
        y_train[i]=cy_train[rndIdx[i]];


    rndIdx=np.random.permutation((np.shape(X_test))[0]); 
    cX_test=np.copy(X_test);
    cy_test=np.copy(y_test);
   
    for i in range(0,np.shape(X_test)[0]):
        X_test[i]=cX_test[rndIdx[i]];
        y_test[i]=cy_test[rndIdx[i]];

    rndIdx=np.random.permutation((np.shape(X_validation))[0]); 
    cX_validation=np.copy(X_validation);
    cy_validation=np.copy(y_validation);
   
    for i in range(0,np.shape(X_validation)[0]):
        X_validation[i]=cX_validation[rndIdx[i]];
        y_validation[i]=cy_validation[rndIdx[i]];
            
    if tgtid:
        return X_train, y_train, X_test, y_test,np.where(rndIdx==466)
    else:
          
        return X_train, y_train, X_test, y_test, X_validation, y_validation


def getCarData2png( path_train='C:/Users/Mir/Documents/MATLAB/DATA aug5/TRAIN/',   path_test='C:/Users/Mir/Documents/MATLAB/DATA aug5/TEST/', path_validation='C:/Users/Mir/Documents/MATLAB/DATA aug5/VALIDATION/',tgtid=False):
    #path_train='C:/0/vc/DATA/TRAIN/';
    #path_test='C:/0/vc/DATA/TEST/';        

    #path_train='C:/Users/Mir/Documents/MATLAB/DATA aug5/TRAIN/'
    #path_test='C:/Users/Mir/Documents/MATLAB/DATA aug5/TEST/'
    #path_validation='C:/Users/Mir/Documents/MATLAB/DATA aug5/VALIDATION/'

    print('Path: '+ path_train)
    print('Collecting Data');
    train_fnamelist=os.listdir(path_train);
    test_fnamelist=os.listdir(path_test);
    validation_fnamelist=os.listdir(path_validation);

    BUS_fnamelist_train=glob.glob(path_train+'BUS/*.png');
    NORMAL_fnamelist_train=glob.glob(path_train+'NORMALCAR/*.png');
    TRUCK_fnamelist_train=glob.glob(path_train+'TRUCK/*.png');
    VAN_fnamelist_train=glob.glob(path_train+'VAN/*.png');

    BUS_fnamelist_test=glob.glob(path_test+'BUS/*.png');
    NORMAL_fnamelist_test=glob.glob(path_test+'NORMALCAR/*.png');
    TRUCK_fnamelist_test=glob.glob(path_test+'TRUCK/*.png');
    VAN_fnamelist_test=glob.glob(path_test+'VAN/*.png');

    BUS_fnamelist_validation=glob.glob(path_validation+'BUS/*.png');
    NORMAL_fnamelist_validation=glob.glob(path_validation+'NORMALCAR/*.png');
    TRUCK_fnamelist_validation=glob.glob(path_validation+'TRUCK/*.png');
    VAN_fnamelist_validation=glob.glob(path_validation+'VAN/*.png');

    X_train = [];
    y_train= np.zeros([len(BUS_fnamelist_train)+
    len(NORMAL_fnamelist_train)+len(TRUCK_fnamelist_train)+len(VAN_fnamelist_train),4])

    X_test=[];
    y_test= np.zeros([len(BUS_fnamelist_test)+
    len(NORMAL_fnamelist_test)+len(TRUCK_fnamelist_test)+len(VAN_fnamelist_test),4])
    
    X_validation = [];
    y_validation= np.zeros([len(BUS_fnamelist_validation)+
    len(NORMAL_fnamelist_validation)+len(TRUCK_fnamelist_validation)+len(VAN_fnamelist_validation),4])

    offset=0;
  
 
    for i in range(offset,len(BUS_fnamelist_train)):
        X_train.append(plt.imread(BUS_fnamelist_train[i]));
        y_train[i+offset][0]=1; #offset=0; just for consistency

        
        
    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_train)):
        X_train.append(plt.imread(NORMAL_fnamelist_train[i]));
        y_train[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_train)):
        X_train.append(plt.imread(TRUCK_fnamelist_train[i]));
        y_train[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_train)):
        X_train.append(plt.imread(VAN_fnamelist_train[i]));
        y_train[i+offset][3]=1;


    offset=0;
    
    for i in range(offset,len(BUS_fnamelist_test)):
        X_test.append(plt.imread(BUS_fnamelist_test[i]));
        y_test[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_test)):
        X_test.append(plt.imread(NORMAL_fnamelist_test[i]));
        y_test[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_test)):
        X_test.append(plt.imread(TRUCK_fnamelist_test[i]));
        y_test[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_test)):
        X_test.append(plt.imread(VAN_fnamelist_test[i]));
        y_test[i+offset][3]=1;


    offset=0;
    
    for i in range(offset,len(BUS_fnamelist_validation)):
        X_validation.append(plt.imread(BUS_fnamelist_validation[i]));
        y_validation[i+offset][0]=1; #offset=0; just for consistency

    offset=offset+i+1;
    for i in range(0,len(NORMAL_fnamelist_validation)):
        X_validation.append(plt.imread(NORMAL_fnamelist_validation[i]));
        y_validation[i+offset][1]=1;
    
    offset=offset+i+1;
    for i in range(0,len(TRUCK_fnamelist_validation)):
        X_validation.append(plt.imread(TRUCK_fnamelist_validation[i]));
        y_validation[i+offset][2]=1;   

    offset=offset+i+1;
    for i in range(0,len(VAN_fnamelist_validation)):
        X_validation.append(plt.imread(VAN_fnamelist_validation[i]));
        y_validation[i+offset][3]=1;

    print('Shuffling...')
    #Shuffle
    
    # np.random.permutation(N): 
    #   list of consecutive integers from 0 to N-1
    #   in a random permutation
  
    rndIdx=np.random.permutation((np.shape(X_train))[0]); 
    cX_train=np.copy(X_train);
    cy_train=np.copy(y_train);

    for i in range(0,np.shape(X_train)[0]):
        X_train[i]=cX_train[rndIdx[i]];
        y_train[i]=cy_train[rndIdx[i]];


    rndIdx=np.random.permutation((np.shape(X_test))[0]); 
    cX_test=np.copy(X_test);
    cy_test=np.copy(y_test);
   
    for i in range(0,np.shape(X_test)[0]):
        X_test[i]=cX_test[rndIdx[i]];
        y_test[i]=cy_test[rndIdx[i]];

    rndIdx=np.random.permutation((np.shape(X_validation))[0]); 
    cX_validation=np.copy(X_validation);
    cy_validation=np.copy(y_validation);
   
    for i in range(0,np.shape(X_validation)[0]):
        X_validation[i]=cX_validation[rndIdx[i]];
        y_validation[i]=cy_validation[rndIdx[i]];
            
    if tgtid:
        return X_train, y_train, X_test, y_test,np.where(rndIdx==466)
    else:
          
        return X_train, y_train, X_test, y_test, X_validation, y_validation



#get i_th batch index range (start to end), given minibatch size and total size
def batch_idx(i,minibatch_size, total_size):
    return list(range(i*minibatch_size,min((i+1)*minibatch_size,total_size)));


imtmp=imread('c.bmp');
imtmp=imtmp.astype('float16')
c=3
dev_im=np.reshape(imtmp[:,:,:],[1,np.shape(imtmp)[0],np.shape(imtmp)[1],c]);