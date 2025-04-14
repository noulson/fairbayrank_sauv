#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

class BPR:

    def __init__(self, sess, dict_args, train_df, vali_df
                 , key_type, user_type_list, item_type_count):
       
        self.dataname = dict_args['dataname']

        self.key_type = key_type
        self.user_type_list = user_type_list
        self.item_type_count = item_type_count

        self.sess = sess
        #self.args = args

        self.num_cols = len(train_df['item_id'].unique())
        self.num_rows = len(train_df['user_id'].unique())

        self.hidden_neuron = dict_args['hidden_neuron']
        self.neg = dict_args['neg']
        self.batch_size = dict_args['batch_size']

        self.train_df = train_df
        self.vali_df = vali_df
        self.num_train = len(self.train_df)
        self.num_vali = len(self.vali_df)

        self.train_epoch = dict_args['train_epoch']

        self.lr = dict_args['lr'] # learning rate
        self.optimizer_method = dict_args['optimizer_method']
        self.display_step = dict_args['display_step']

        #self.num_genre = dict_args['num_type']

        self.reg = dict_args['reg'] # regularization term trade-off

        print('**********BPR**********')
        #print(self.args)
        self._prepare_model()

    def run(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(1, self.train_epoch + 1):
            self.train_model(epoch_itr)
            if epoch_itr % self.display_step == 0:
                self.test_model(epoch_itr)
        return self.make_records()

    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

        with tf.variable_scope("BPR", reuse=tf.AUTO_REUSE):
            self.P = tf.get_variable(name="P", initializer=tf.truncated_normal(shape=[self.num_rows, self.hidden_neuron],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)
            self.Q = tf.get_variable(name="Q", initializer=tf.truncated_normal(shape=[self.num_cols+1, self.hidden_neuron],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)

        self.saver = tf.train.Saver([self.P, self.Q])

        p = tf.reduce_sum(tf.nn.embedding_lookup(self.P, self.user_input), 1)
        q_neg = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_neg), 1)
        q_pos = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_pos), 1)

        predict_pos = (tf.reduce_sum(p * q_pos, 1))
        predict_neg = (tf.reduce_sum(p * q_neg, 1))

        cost1 = tf.reduce_sum(tf.nn.softplus(-(predict_pos - predict_neg)))
        cost2 = self.reg * 0.5 * (self.l2_norm(self.P) + self.l2_norm(self.Q))  # regularization term

        self.cost = cost1 + cost2  # the loss function

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.optimizer = optimizer.minimize(self.cost)

    def train_model(self, itr):
        NS_start_time = time.time() * 1000.0
        epoch_cost = 0
        num_sample, user_list, item_pos_list, item_neg_list = utility.negative_sample(self.train_df, self.num_rows,
                                                                                      self.num_cols, self.neg)
        NS_end_time = time.time() * 1000.0

        start_time = time.time() * 1000.0
        num_batch = int(len(user_list) / float(self.batch_size)) + 1
        random_idx = np.random.permutation(len(user_list))
        for i in range(num_batch):

            # get the indices of the current batch
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
            _, tmp_cost = self.sess.run(  # do the optimization by the minibatch
                [self.optimizer, self.cost],
                feed_dict={self.user_input: user_list[batch_idx, :],
                           self.item_input_pos: item_pos_list[batch_idx, :],
                           self.item_input_neg: item_neg_list[batch_idx, :]})
            epoch_cost += tmp_cost

        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
                   "Training time : %d ms" % (time.time() * 1000.0 - start_time),
                   "negative Sampling time : %d ms" % (NS_end_time - NS_start_time),
                   "negative samples : %d" % (num_sample))

        ckpt_save_path = "./"+self.dataname+"/BPR_check_points"
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        self.saver.save(sess, ckpt_save_path + "/check_point.ckpt", global_step=itr)

    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        if itr % self.display_step == 0:
            start_time = time.time() * 1000.0
            P, Q = self.sess.run([self.P, self.Q])
            Rec = np.matmul(P, Q.T)

            [precision, recall, f_score, NDCG] = utility.test_model_all(Rec, self.vali_df, self.train_df)
#             utility.ranking_analysis(Rec, self.vali_df, self.train_df, self.key_type, self.user_type_list,
#                                      self.item_type_count)
            utility.test_model_per_user_type(Rec, self.vali_df, self.train_df, self.user_type_list, self.key_type)
            auc_global = utility.auc_per_user(Rec, self.vali_df, self.train_df)

            print("AUC global is: ", auc_global)
            
#             for k in self.key_type:
#                 print("AUC per %d is:\t[%.7f] "%(k, auc[k]))
#             print("AUC global is: ", auc_global)
            print (
                "Testing //", "Epoch %d //" % itr,
                "Testing time : %d ms" % (time.time() * 1000.0 - start_time))
            print("=" * 200)
        

            filename = './unfairBPR_results/epoch'+ str(itr) +'_Rec_' + self.dataname + '_BPR.npy'
            os.makedirs(os.path.dirname(filename), exist_ok=True)           
            with open(filename, "wb") as f:
                np.save(f, Rec)



    def make_records(self):  # record all the results' details into files
        P, Q = self.sess.run([self.P, self.Q])
        Rec = np.matmul(P, Q.T)

        [precision, recall, f_score, NDCG] = utility.test_model_all(Rec, self.vali_df, self.train_df)
        return precision, recall, f_score, NDCG, Rec

    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))


# In[5]:


#optimizer_method = ['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent','Momentum'], default='Adam')


train_epoch = 41
display_step = 1
lr = 0.01
reg = 0.1
optimizer_method = 'Adam'
hidden_neuron = 20
n = 1
neg = 5
batch_size = 256
dataname = 'ml1m-6'


# In[6]:


dict_args =  {"train_epoch": train_epoch,
            "display_step":display_step,
            "lr":lr,
            "reg":reg,
            "optimizer_method":optimizer_method,
            "hidden_neuron":hidden_neuron,
            "n":n,
            "neg":neg,
            "batch_size":batch_size,
            "dataname":dataname}
dict_args


# In[7]:


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


# In[8]:


train_df['item_id'].unique()


# In[9]:


train_df.head(20)


# In[10]:


train_df.shape


# In[11]:


test_df.head(20)


# In[12]:


test_df.shape


# In[13]:


print(len(user_idd_type_list))


# In[14]:


user_idd_type_list


# In[ ]:





# In[15]:


print(type_user_vector['F'].shape)


# In[16]:


type_user_vector


# In[17]:


len(item_type_count)


# In[18]:


item_type_count


# In[19]:


print(type_count)


# In[20]:


num_item = len(train_df['item_id'].unique())
num_user = len(train_df['user_id'].unique())
num_type = len(key_type)
print('items number : ',num_item)
print('users number : ',num_user)
print('user types : ',key_type)


# In[21]:


dict_args["num_type"] = len(key_type)


# In[22]:


user_type_list = [] #preprocessing to be sure that user types are really the right ones armielle 
for u in range(num_user):
    gl = user_idd_type_list[u]
    tmp = []
    for g in gl:
        if g in key_type:
            tmp.append(g)
    user_type_list.append(tmp)

print(len(user_type_list))


# In[23]:


len(user_type_list)


# In[24]:


print('*' * 50)
print('number of positive feedback: ' + str(len(train_df)))
print('estimated number of training samples: ' + str(neg * len(train_df)))
print('*' * 50)


# In[25]:


# generate user_type matrix
type_user_indicator = np.zeros((num_type, num_user))

for k in range(num_type):
    type_user_indicator[k,:] = type_user_vector[key_type[k]]


# In[26]:


precision = np.zeros(4)
recall = np.zeros(4)
f1 = np.zeros(4)
ndcg = np.zeros(4)
RSP = np.zeros(4)
REO = np.zeros(4)

precision 


# In[27]:


len(user_type_list)


# In[28]:


train_df['item_id'].unique()


# In[27]:


tf.compat.v1.disable_eager_execution()

for i in range(n):
    with tf.compat.v1.Session() as sess:
        bpr = BPR(sess, dict_args, train_df, test_df, key_type, user_type_list, item_type_count)
        [prec_one, rec_one, f_one, ndcg_one, Rec] = bpr.run()
        #[RSP_one, REO_one] = utility.ranking_analysis(Rec, vali_df, train_df, key_genre, item_genre_list, user_genre_count)
#         precision += prec_one
#         recall += rec_one
#         f1 += f_one
#         ndcg += ndcg_one
#         RSP += RSP_one
#         REO += REO_one


# In[54]:


list_recom_unfairbpr = []
list_recom_constiemfairadvbpr = []
list_recom_fairadvbpr = []
list_recom_iemfairadvbpr = []
list_recom_uemfairbpr = []

for itr in range(1, train_epoch + 1):
    filename = './unfairBPR_results/epoch'+ str(itr) +'_Rec_' + dataname + '_BPR.npy'
    with open(filename, "rb") as f:
        recom = np.load(f)
        list_recom_unfairbpr.append(recom)

for itr in range(2, train_epoch + 2):
    filename1 = './const_IembfairAdvBPR_results/epoch'+ str(itr) +'_Rec_' + dataname + '_constfairAdvBPR.npy'
    with open(filename1, "rb") as f:
        recom = np.load(f)
        list_recom_constiemfairadvbpr.append(recom)
    
    filename2 = './fairAdvBPR_results/epoch'+ str(itr) +'_Rec_' + dataname + '_fairAdvBPR.npy'
    with open(filename2, "rb") as f:
        recom = np.load(f)
        list_recom_fairadvbpr.append(recom)
        
    filename3 = './IembfairAdvBPR_results/epoch'+ str(itr) +'_Rec_' + dataname + '_iembfairAdvBPR.npy'
    with open(filename3, "rb") as f:
        recom = np.load(f)
        list_recom_iemfairadvbpr.append(recom)
        
    filename4 = './UembfairAdvBPR_results/epoch'+ str(itr) +'_Rec_' + dataname + '_uembfairAdvBPR.npy'
    with open(filename4, "rb") as f:
        recom = np.load(f)
        list_recom_uemfairbpr.append(recom)



        


# In[46]:


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
    
def plotline_save_as_pdf(title='',axis_x = None, xlabel = '', axis_y1 = None, axis_y2 = None , axis_y3 = None, axis_y4 = None,axis_y5 = None, ylabel ='', linewidth=3, plot_file = '',legendy1='',legendy2='',legendy3='',legendy4='',legendy5=''):
    if axis_y1 is not None:
        plt.plot(axis_x, axis_y1, color='orange', linewidth = linewidth, label = legendy1)
    if axis_y2 is not None:
        plt.plot(axis_x, axis_y2, 'g', linewidth = linewidth, label = legendy2)
    if axis_y3 is not None:
        plt.plot(axis_x, axis_y3, 'b', linewidth = linewidth, label = legendy3)
    if axis_y4 is not None:
        plt.plot(axis_x, axis_y4, 'black', linewidth = linewidth, label = legendy4)
    if axis_y5 is not None:
        plt.plot(axis_x, axis_y5, 'red', linewidth = linewidth, label = legendy5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(plot_file)


# In[55]:


auc_plot_unfairbpr=[]
auc_plot_constiemfairbpr=[]
auc_plot_fairadvbpr=[]
auc_plot_iemfairadvbpr=[]
auc_plot_uemfairadvbpr=[]

for rec in list_recom_unfairbpr:
    [precision, recall, f_score, NDCG] = utility.test_model_all(rec, test_df, train_df)

    utility.test_model_per_user_type(rec, test_df, train_df, user_type_list, key_type)
    auc = utility.auc_per_user(rec, test_df, train_df)
    auc_plot_unfairbpr.append(auc)
    
for rec in list_recom_constiemfairadvbpr:
    [precision, recall, f_score, NDCG] = utility.test_model_all(rec, test_df, train_df)

    utility.test_model_per_user_type(rec, test_df, train_df, user_type_list, key_type)
    auc = utility.auc_per_user(rec, test_df, train_df)
    auc_plot_constiemfairbpr.append(auc)

for rec in list_recom_fairadvbpr:
    [precision, recall, f_score, NDCG] = utility.test_model_all(rec, test_df, train_df)

    utility.test_model_per_user_type(rec, test_df, train_df, user_type_list, key_type)
    auc = utility.auc_per_user(rec, test_df, train_df)
    auc_plot_fairadvbpr.append(auc)

for rec in list_recom_iemfairadvbpr:
    [precision, recall, f_score, NDCG] = utility.test_model_all(rec, test_df, train_df)

    utility.test_model_per_user_type(rec, test_df, train_df, user_type_list, key_type)
    auc = utility.auc_per_user(rec, test_df, train_df)
    auc_plot_iemfairadvbpr.append(auc)
    
for rec in list_recom_uemfairbpr:
    [precision, recall, f_score, NDCG] = utility.test_model_all(rec, test_df, train_df)

    utility.test_model_per_user_type(rec, test_df, train_df, user_type_list, key_type)
    auc = utility.auc_per_user(rec, test_df, train_df)
    auc_plot_uemfairadvbpr.append(auc)
  


# In[59]:


plotline_save_as_pdf(title='AUC On Testing',axis_x = range(1,train_epoch+1), xlabel = 'Epoch', axis_y1 = auc_plot_unfairbpr, axis_y2 = auc_plot_constiemfairbpr , axis_y3 = auc_plot_fairadvbpr,axis_y4 = auc_plot_iemfairadvbpr,axis_y5 = auc_plot_uemfairadvbpr, ylabel ='AUC', linewidth=3, plot_file = './unfairBPR_plots/auc_unfairBpr.pdf',legendy1='UnfairBPR',legendy2='CFairAdvBPR',legendy3='fairadv',legendy4='iemfairadv',legendy5='uemfairadv')
  


# In[31]:


filename = './unfairBPR_plots' 


# In[ ]:




