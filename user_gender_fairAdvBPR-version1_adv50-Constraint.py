#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import time
import numpy as np
import os
import copy
import pickle
import argparse
import utilityarm as utility
import pandas as pd
from sklearn.metrics import *
import tensorflow.keras.backend as K


# In[2]:


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

class FairAdvBPR:

    def __init__(self, sess, dict_args, train_df, test_df, user_type, type_error_weight, key_type, user_type_list, item_type_count):
       
        self.dataname = dict_args['dataname']

        self.key_type = key_type
        self.user_type_list = user_type_list
        self.item_type_count = item_type_count
        self.layers = dict_args['layers']
        self.sess = sess
        
        self.num_cols = len(train_df['item_id'].unique())
        self.num_rows = len(train_df['user_id'].unique())

        self.hidden_neuron = dict_args['hidden_neuron']
        self.neg = dict_args['neg']
        self.batch_size = dict_args['batch_size']

        self.train_df = train_df
        self.vali_df = test_df
        self.num_train = len(self.train_df)
        self.num_vali = len(self.vali_df)

        self.train_epoch = dict_args['train_epoch']
        self.train_epoch_a = dict_args['train_epoch_a']

        self.lr_r = dict_args['lr_r'] # learning rate
        self.lr_a = dict_args['lr_a'] # learning rate
        self.alpha = dict_args['alpha'] # learning rate
        self.optimizer_method = dict_args['optimizer_method']
        self.display_step = dict_args['display_step']
        
        self.type_error_weight = type_error_weight
        self.num_type = dict_args['num_type']
        
        self.user_type = user_type
        self.type_count_list = []
        for k in range(self.num_type):
            self.type_count_list.append(np.sum(user_type[:,k]))

        
        self.reg = dict_args['reg'] # regularization term trade-off
        self.reg_s = dict_args['reg_s']

        print('**********fairAdvBPR**********')
        #print(self.args)
        self._prepare_model()

    def loadmodel(self, saver, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        
    def run(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        saver = tf.train.Saver([self.P, self.Q])
        self.loadmodel(saver, "./"+self.dataname+"/BPR_check_points")

        for epoch_itr in range(1, self.train_epoch + 1 + self.train_epoch_a):
            self.train_model(epoch_itr)
            if epoch_itr % self.display_step == 0:
                self.test_model(epoch_itr)
        return self.make_records()

    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

            self.input_user_type = tf.placeholder(dtype=tf.float32, shape=[None, self.num_type]
                                                   , name="input_user_type")
            self.input_user_error_weight = tf.placeholder(dtype=tf.float32, shape=[None, 1]
                                                          , name="input_user_error_weight")

        with tf.variable_scope("BPR", reuse=tf.AUTO_REUSE):
            self.P = tf.get_variable(name="P",
                                     initializer=tf.truncated_normal(shape=[self.num_rows, self.hidden_neuron], mean=0,
                                                                     stddev=0.03), dtype=tf.float32)
            self.Q = tf.get_variable(name="Q",
                                     initializer=tf.truncated_normal(shape=[self.num_cols+1, self.hidden_neuron], mean=0,
                                                                     stddev=0.03), dtype=tf.float32)
        para_r = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="BPR")

        with tf.variable_scope("Adversarial", reuse=tf.AUTO_REUSE):
            num_layer = len(self.layers)
            adv_W = []
            adv_b = []
            for l in range(num_layer):
                if l == 0:
                    in_shape = 21
                else:
                    in_shape = self.layers[l - 1]
                adv_W.append(tf.get_variable(name="adv_W" + str(l),
                                             initializer=tf.truncated_normal(shape=[in_shape, self.layers[l]],
                                                                             mean=0, stddev=0.03), dtype=tf.float32))
                adv_b.append(tf.get_variable(name="adv_b" + str(l),
                                             initializer=tf.truncated_normal(shape=[1, self.layers[l]],
                                                                             mean=0, stddev=0.03), dtype=tf.float32))
            adv_W_out = tf.get_variable(name="adv_W_out",
                                        initializer=tf.truncated_normal(shape=[self.layers[-1], self.num_type],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)

            adv_b_out = tf.get_variable(name="adv_b_out",
                                        initializer=tf.truncated_normal(shape=[1, self.num_type],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        para_a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adversarial")

        p = tf.reduce_sum(tf.nn.embedding_lookup(self.P, self.user_input), 1)
        q_neg = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_neg), 1)
        q_pos = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_pos), 1)

        predict_pos = tf.reduce_sum(p * q_pos, 1)
        predict_neg = tf.reduce_sum(p * q_neg, 1)

        r_cost1 = tf.reduce_sum(tf.nn.softplus(-(predict_pos - predict_neg)))
        r_cost2 = self.reg * 0.5 * (self.l2_norm(self.P) + self.l2_norm(self.Q))  # regularization term
        pred = tf.matmul(self.P, tf.transpose(self.Q))
        self.s_mean = tf.reduce_mean(pred, axis=1)
        self.s_std = tf.keras.backend.std(pred, axis=1)
        self.s_cost = tf.reduce_sum(tf.square(self.s_mean) + tf.square(self.s_std) - 2 * tf.log(self.s_std) - 1)#additional regularization
        
        
#        print('input_user_type ',self.input_user_type[:,1].shape)
#         usertype = self.input_user_type[]
#         print('shape',usertype.shape)
        const_user_type1 = tf.reduce_sum(tf.nn.softplus(-(predict_pos - predict_neg) * self.input_user_type[:,0]))
        const_user_type2 = tf.reduce_sum(tf.nn.softplus(-(predict_pos - predict_neg) * self.input_user_type[:,1]))
        
        const = K.sqrt(K.square(const_user_type1 - const_user_type2))
        
        self.r_cost = r_cost1 + r_cost2 + 0.5 * const + self.reg_s * 0.5 * self.s_cost
        
        print('shape q pos',q_pos.shape)
        print('shape p',p.shape)
        print('shape predict pos ', predict_pos.shape)
        
        adv_last = tf.reshape(predict_pos, [tf.shape(self.input_user_type)[0], 1])
        print('shape adv_last ', adv_last.shape)
        adv_last = tf.concat([adv_last, q_pos], 1)
        
        for l in range(num_layer):
            adv = tf.nn.relu(tf.matmul(adv_last, adv_W[l]) + adv_b[l])
            adv_last = adv
        self.adv_output = tf.nn.sigmoid(tf.matmul(adv_last, adv_W_out) + adv_b_out)
        self.a_cost = tf.reduce_sum(tf.square(self.adv_output - self.input_user_type) * self.input_user_error_weight)

        self.all_cost = self.r_cost - self.alpha * self.a_cost  # the loss function

        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.r_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.r_cost, var_list=para_r)
            self.a_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_a).minimize(self.a_cost, var_list=para_a)
            self.all_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.all_cost, var_list=para_r)


    def train_model(self, itr):
        NS_start_time = time.time() * 1000.0
        epoch_r_cost = 0.0
        epoch_s_cost = 0.0
        epoch_s_mean = 0.0
        epoch_s_std = 0.0
        epoch_a_cost = 0.0
        num_sample, user_list, item_pos_list, item_neg_list = utility.negative_sample(self.train_df, self.num_rows,
                                                                                      self.num_cols, self.neg)
        NS_end_time = time.time() * 1000.0

        start_time = time.time() * 1000.0
        num_batch = int(num_sample / float(self.batch_size)) + 1
        random_idx = np.random.permutation(num_sample)
        for i in range(num_batch):
            # get the indices of the current batch
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

            if itr > self.train_epoch:
                random_idx_a = np.random.permutation(num_sample)
                print("boucle adversarial debut-- num batch ",i)
                for j in range(num_batch):
                    if j == num_batch - 1:
                        batch_idx_a = random_idx_a[j * self.batch_size:]
                    elif j < num_batch - 1:
                        batch_idx_a = random_idx_a[(j * self.batch_size):((j + 1) * self.batch_size)]
                    user_idx_list = ((user_list[batch_idx_a, :]).reshape((len(batch_idx_a)))).tolist()
                    _, tmp_a_cost = self.sess.run(  # do the optimization by the minibatch
                        [self.a_optimizer, self.a_cost],
                        feed_dict={self.user_input: user_list[batch_idx_a, :],
                                   self.item_input_pos: item_pos_list[batch_idx_a, :],
                                   self.item_input_neg: item_neg_list[batch_idx_a, :],
                                   self.input_user_type: self.user_type[user_idx_list,:],
                                   self.input_user_error_weight: self.type_error_weight[user_idx_list,:]})
                    epoch_a_cost += tmp_a_cost

                user_idx_list = ((user_list[batch_idx, :]).reshape((len(batch_idx)))).tolist()
                _, tmp_r_cost, tmp_s_cost, tmp_s_mean, tmp_s_std = self.sess.run(  # do the optimization by the minibatch
                    [self.all_optimizer, self.all_cost, self.s_cost, self.s_mean, self.s_std],
                    feed_dict={self.user_input: user_list[batch_idx, :],
                               self.item_input_pos: item_pos_list[batch_idx, :],
                               self.item_input_neg: item_neg_list[batch_idx, :],
                               self.input_user_type: self.user_type[user_idx_list, :],
                               self.input_user_error_weight: self.type_error_weight[user_idx_list, :]})
                epoch_r_cost += tmp_r_cost
                epoch_s_mean += np.mean(tmp_s_mean)
                epoch_s_std += np.mean(tmp_s_std)
                epoch_s_cost += tmp_s_cost
                print("boucle adversarial fin")
            else:
                user_idx_list = ((user_list[batch_idx, :]).reshape((len(batch_idx)))).tolist()
                _, tmp_r_cost, tmp_s_cost, tmp_s_mean, tmp_s_std = self.sess.run(  # do the optimization by the minibatch
                    [self.r_optimizer, self.r_cost, self.s_cost, self.s_mean, self.s_std],
                    feed_dict={self.user_input: user_list[batch_idx, :],
                               self.item_input_pos: item_pos_list[batch_idx, :],
                               self.item_input_neg: item_neg_list[batch_idx, :],
                               self.input_user_type: self.user_type[user_idx_list, :],
                               self.input_user_error_weight: self.type_error_weight[user_idx_list, :]})
                epoch_r_cost += tmp_r_cost
                epoch_s_mean += np.mean(tmp_s_mean)
                epoch_s_std += np.mean(tmp_s_std)
                epoch_s_cost += tmp_s_cost
        epoch_a_cost /= num_batch
        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % itr, " Total r_cost = %.5f" % epoch_r_cost,
                   " Total s_cost = %.5f" % epoch_s_cost,
                   " Total s_mean = %.5f" % epoch_s_mean,
                   " Total s_std = %.5f" % epoch_s_std,
                   " Total a_cost = %.5f" % epoch_a_cost,
                   "Training time : %d ms" % (time.time() * 1000.0 - start_time),
                   "negative Sampling time : %d ms" % (NS_end_time - NS_start_time),
                   "negative samples : %d" % (num_sample))
       
    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        if itr % self.display_step == 0:
            start_time = time.time() * 1000.0
            P, Q = self.sess.run([self.P, self.Q])
            Rec = np.matmul(P, Q.T)

            [precision, recall, f_score, NDCG] = utility.test_model_all(Rec, self.vali_df, self.train_df)
#             utility.ranking_analysis(Rec, self.vali_df, self.train_df, self.key_genre, self.item_genre_list,
#                                      self.user_genre_count)
            utility.test_model_per_user_type(Rec, self.vali_df, self.train_df, self.user_type_list, self.key_type)
            auc = utility.auc_per_user(Rec, self.vali_df, self.train_df)
            print("AUC global is: ", auc)
            
            filename = './const_IembfairAdvBPR_adv50_results_bis/epoch'+ str(itr) +'_Rec_' + self.dataname + '_constfairAdvBPR.npy'
            os.makedirs(os.path.dirname(filename), exist_ok=True)           
            with open(filename, "wb") as f:
                np.save(f, Rec)
            

    def make_records(self):  # record all the results' details into files
        P, Q = self.sess.run([self.P, self.Q])
        Rec = np.matmul(P, Q.T)

        [precision, recall, f_score, NDCG] = utility.test_model_all(Rec, self.vali_df, self.train_df)
        return precision, recall, f_score, NDCG, Rec

#     def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
#         if itr % self.display_step == 0:
#             start_time = time.time() * 1000.0
#             P, Q = self.sess.run([self.P, self.Q])
#             Rec = np.matmul(P, Q.T)

#             [precision, recall, f_score, NDCG] = utility.test_model_all(Rec, self.vali_df, self.train_df)
# #             utility.ranking_analysis(Rec, self.vali_df, self.train_df, self.key_type, self.user_type_list,
# #                                      self.item_type_count)
#             utility.test_model_per_user_type(Rec, self.vali_df, self.train_df, self.user_type_list, self.key_type)
#             auc = utility.auc_per_user(Rec, self.vali_df, self.train_df)
#             print("AUC global is: ", auc)
#             print (
#                 "Testing //", "Epoch %d //" % itr,
#                 "Testing time : %d ms" % (time.time() * 1000.0 - start_time))
#             print("=" * 200)


    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))


# In[3]:


#optimizer_method = ['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent','Momentum'], default='Adam')


train_epoch = 1
train_epoch_a = 50 #default 20
display_step = 1
lr_r = 0.01
lr_a = 0.005
reg = 0.1
reg_s = 30
alpha = 1000
optimizer_method = 'Adam'
hidden_neuron = 20
n = 1
neg = 5
batch_size = 256
layers = [50, 50, 50, 50]
dataname = 'ml1m-6'


# In[4]:


dict_args =  {"train_epoch": train_epoch,
              "train_epoch_a": train_epoch_a,
            "display_step":display_step,
            "lr_r":lr_r,
            "lr_a":lr_a,
            "reg":reg,
            "reg_s":reg_s,
            "alpha":alpha,
            "optimizer_method":optimizer_method,
            "hidden_neuron":hidden_neuron,
            "n":n,
            "neg":neg,
            "batch_size":batch_size,
            "layers":layers,
            "dataname":dataname}
dict_args


# In[5]:


with open('./training_df.pkl', 'rb') as f:
    train_df = pickle.load(f,encoding='latin1')

# with open('./' + dataname + '/valiing_df.pkl', 'rb') as f:
#     vali_df = pickle.load(f,encoding='latin1')  # for validation
    
with open('./testing_df.pkl', 'rb') as f:
    test_df = pickle.load(f,encoding='latin1')  # for validation
# vali_df = pickle.load(open('./' + dataname + '/testing_df.pkl'))  # for testing

with open('./key_type.pkl', 'rb') as f:
    key_type = pickle.load(f,encoding='latin1')
    
with open('./user_idd_type_list.pkl', 'rb') as f:
    user_idd_type_list = pickle.load(f,encoding='latin1')
    
with open('./type_user_vector.pkl', 'rb') as f:
    type_user_vector = pickle.load(f,encoding='latin1')

with open('./type_count.pkl', 'rb') as f:
    type_count = pickle.load(f,encoding='latin1')
    
with open('./item_type_count.pkl', 'rb') as f:
    item_type_count = pickle.load(f,encoding='latin1')


# In[6]:


train_df.head(20)


# In[7]:


train_df.shape


# In[8]:


test_df.head(20)


# In[9]:


test_df.shape


# In[10]:


print(len(user_idd_type_list))


# In[11]:


user_idd_type_list


# In[ ]:





# In[12]:


print(type_user_vector['F'].shape)


# In[13]:


type_user_vector


# In[14]:


len(item_type_count)


# In[15]:


item_type_count


# In[16]:


print(type_count)


# In[17]:


num_item = len(train_df['item_id'].unique())
num_user = len(train_df['user_id'].unique())
num_type = len(key_type)
print('items number : ',num_item)
print('users number : ',num_user)
print('user types : ',key_type)


# In[18]:


dict_args["num_type"] = len(key_type)


# In[19]:


user_type_list = [] #preprocessing to be sure that user types are really the right ones armielle 
for u in range(num_user):
    gl = user_idd_type_list[u]
    tmp = []
    for g in gl:
        if g in key_type:
            tmp.append(g)
    user_type_list.append(tmp)

print(len(user_type_list))


# In[20]:


# genreate user_type matrix
user_type = np.zeros((num_user, num_type))
for u in range(num_user):
    gl = user_type_list[u]
    for k in range(num_type):
        if key_type[k] in gl:
            user_type[u, k] = 1.0


# In[21]:


len(user_type_list)


# In[22]:


print('*' * 50)
print('number of positive feedback: ' + str(len(train_df)))
print('estimated number of training samples: ' + str(neg * len(train_df)))
print('*' * 50)


# In[23]:


type_count_mean_reciprocal = []
for k in key_type:
    type_count_mean_reciprocal.append(1.0 / type_count[k])
type_count_mean_reciprocal = (np.array(type_count_mean_reciprocal)).reshape((num_type, 1))
type_error_weight = np.dot(user_type, type_count_mean_reciprocal)


# In[24]:


# generate user_type matrix
type_user_indicator = np.zeros((num_type, num_user))

for k in range(num_type):
    type_user_indicator[k,:] = type_user_vector[key_type[k]]


# In[25]:


precision = np.zeros(4)
recall = np.zeros(4)
f1 = np.zeros(4)
ndcg = np.zeros(4)
RSP = np.zeros(4)
REO = np.zeros(4)

precision 


# In[26]:


len(user_type)


# In[27]:


user_type


# In[28]:


tf.compat.v1.disable_eager_execution()

for i in range(n):
    with tf.compat.v1.Session() as sess:
        fairadvbpr = FairAdvBPR(sess, dict_args, train_df, test_df, user_type, type_error_weight, key_type, user_type_list, item_type_count)
        [prec_one, rec_one, f_one, ndcg_one, Rec] = fairadvbpr.run()
        #[RSP_one, REO_one] = utility.ranking_analysis(Rec, vali_df, train_df, key_genre, item_genre_list, user_genre_count)
#         precision += prec_one
#         recall += rec_one
#         f1 += f_one
#         ndcg += ndcg_one
#         RSP += RSP_one
#         REO += REO_one


# In[ ]:


# with open('Rec_' + dataname + '_fairAdvBPR.npy', "wb") as f:
#     np.save(f, Rec)


# In[ ]:


# with open('Rec_' + dataname + '_fairAdvBPR.npy', "rb") as f:
#     Recom = np.load(f)
# Recom.shape


# In[ ]:


# [precision, recall, f_score, NDCG] = utility.test_model_all(Recom, test_df, train_df)

# utility.test_model_per_user_type(Recom, test_df, train_df, user_type_list, key_type)
# auc = utility.auc_per_user(Recom, test_df, train_df)
# print("AUC global is: ", auc)


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
from colour import Color

def savepdf_barplot_color_gradient(ymin = 0.5, ymax = 0.7, whis = 5, start_color='pink',end_color='blue',num_color=5, title='',axis_x = None, xlabel = '', axis_y1 = None, ylabel ='', plot_file = ''):
    
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    (ax1, ax2) = gs.subplots(sharex='col', sharey='row')
    
    red = Color(start_color)
    colors = list(red.range_to(Color(end_color), num_color))
    colors = [color.rgb for color in colors]
    
    X_axis = np.arange(len(axis_x))
    ax1.bar(X_axis, axis_y1, color=colors)
    ax1.hlines(y=axis_y1[0], xmin = 0, xmax = len(axis_x)-1, colors='black', linestyles='--', lw=1)
    
    plt.sca(ax1)
    plt.xticks(X_axis, axis_x, rotation =50)
    #plt.xlabel(xlabel)
    #fig.suptitle(title)
    plt.ylabel(ylabel, fontsize=18)
    plt.rcParams.update({'font.size': 13}) 
    plt.grid()
    
    plt.sca(ax2)
    ax2.boxplot(axis_y1, whis = whis)
    ax1.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    plt.savefig(plot_file)

def savepdf_barplot_color_gradient2(start_color='pink',end_color='blue',num_color=20, title='',axis_x = None, xlabel = '', axis_y1 = None, ylabel ='', plot_file = ''):
    
    red = Color(start_color)
    colors = list(red.range_to(Color(end_color), num_color))
    colors = [color.rgb for color in colors]
    
    X_axis = np.arange(len(axis_x))
    plt.bar(X_axis, axis_y1, color=colors)
    
    
    plt.xticks(X_axis, axis_x, rotation =70)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    
   # plt.tight_layout()
    plt.savefig(plot_file)

