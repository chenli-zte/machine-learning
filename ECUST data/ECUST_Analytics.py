
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rc('figure', figsize=(15, 6))
from dateutil.parser import parse


# ### 读取 CPU\磁盘读\磁盘写\网络出口\网络入口\内存 数据

# In[2]:

vCpuUsage = pd.read_excel('../ECUST data/Guangxi university data 20161111/CPU_20161111105339.xlsx',converters={u'时间':parse})
vDiskRead = pd.read_excel('../ECUST data/Guangxi university data 20161111/DiskRead_20161111110123.xlsx',converters={u'时间':parse})
vDiskWrite = pd.read_excel('../ECUST data/Guangxi university data 20161111/DiskWrite_20161111110158.xlsx',converters={u'时间':parse})
vNwEgress = pd.read_excel('../ECUST data/Guangxi university data 20161111/NetworkEgress_20161111110304.xlsx',converters={u'时间':parse})
vNwIngress = pd.read_excel('../ECUST data/Guangxi university data 20161111/NetworkIngress_20161111110416.xlsx',converters={u'时间':parse})
vMemUsage = pd.read_excel('../ECUST data/Guangxi university data 20161111/Memory_20161111105932.xlsx',converters={u'时间':parse})


# ### 修改index和columns

# In[3]:

for var in (vCpuUsage,vDiskRead,vDiskWrite,vNwEgress,vNwIngress,vMemUsage):
    var.rename(columns={u'资源':'vres', u'类型':'vtype',u'时间':'vtime', u'最大值':'vmax',
                        u'最小值':'vmin', u'平均值':'vavg', u'单位':'vunit'}, inplace = True)
    if 'vtime' in var.columns.values:
        var.set_index('vtime',inplace=True) 


# ### 初步探索时间序列数据,形成待分析多维数据矩阵X

# In[4]:

for var in (vCpuUsage,vDiskRead,vDiskWrite,vNwEgress,vNwIngress,vMemUsage):
    print(var.head())
    print(var.count())


# In[5]:

vDiskRead.to_period('Min').head()


# In[6]:

X = pd.concat([vCpuUsage.to_period('Min').vavg,
               vDiskRead.to_period('Min').vavg,
               vDiskWrite.to_period('Min').vavg,
               vNwEgress.to_period('Min').vavg,
               vNwIngress.to_period('Min').vavg,
               vMemUsage.to_period('Min').vavg],axis=1,keys=['vCpuUsage','vDiskRead','vDiskWrite','vNwEgress','vNwIngress','vMemUsage'])
X.head()


# In[7]:

#设定初始值后，对NaN进行线性插值
X.ix[0,X.ix[0].isnull()]=0
X.interpolate(method='time',inplace=True)
X.head()


# In[8]:

X_mean=X.mean()
X_std=X.std()
X.describe()


# ### 数据预处理-标准化

# In[9]:

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[111]:

X_scaler=StandardScaler().fit(X)
#scikit learn操作的数据类型为ndarray，因此再次转换为DataFrame
X = DataFrame(X_scaler.transform(X),index=X.index,columns=X.columns)
X_mean=X.mean()
X_std=X.std()
print('std var:\n',X_std,'\n\n','mean:\n',X_mean)


# ### 多维数据探索

# In[11]:

plt.subplot(611)
X.vCpuUsage.plot(label='vCpuUsage')
plt.legend(loc='best')
plt.subplot(612)
X.vDiskRead.plot(label='DiskRead')
plt.legend(loc='best')
plt.subplot(613)
X.vDiskWrite.plot(label='DiskWirte')
plt.legend(loc='best')
plt.subplot(614)
X.vNwEgress.plot(label='NwEgress')
plt.legend(loc='best')
plt.subplot(615)
X.vNwIngress.plot(label='NwIngress')
plt.legend(loc='best')
plt.subplot(616)
X.vMemUsage.plot(label='MemUsage')
plt.legend(loc='best')

pd.scatter_matrix(X,diagonal='kde',color='k',alpha=0.3,figsize=(15,9))


# >直观感觉：1）特征之间有相关性；2）特征的高斯性不明显；3）Nw I/O在已知的故障时刻变化不明显

# ### PCA

# In[12]:

from sklearn.decomposition import PCA
import numpy.linalg as nlg


# #### PCA数据探索

# In[66]:

X_pca=PCA().fit(X)


# In[73]:

print('covariance matrix is:\n',X_pca.get_covariance())
print('\nexplained_variance is: \n',X_pca.explained_variance_)
print('\nexplained_variance ratio is: \n',X_pca.explained_variance_ratio_)


# >* NwEgress,NwIngress有较强的线性相关性(0.83419459)，DiskRead,DiskWrite的相关性也不低（0.48996776），另外，MemUsage与Nw I/O也都有一定的相关性（0.31194841，0.27734822），为避免各个特征的相互干扰，应去除相关性
# * 通过PCA进行特征转换后，某些分量对应的特征值远小于其他值，说明该分量对于样本区分几乎无贡献，出于降维考虑可丢弃

# In[13]:

#指定主成分的方差和所占的最小比例阈值为0.85
X_pca=PCA(n_components=0.85).fit(X)
print(X_pca.components_)


# In[14]:

print(X.ix[0].shape,X_pca.components_.shape,X_pca.explained_variance_.shape,len(X))


# In[15]:

X_pca_recover=DataFrame(np.dot(X,np.dot(X_pca.components_.T,X_pca.components_)),index=X.index,columns=X.columns)


# In[16]:

plt.subplot(611)
X_pca_recover.vCpuUsage.plot(label='vCpuUsage_ica')
plt.legend(loc='best')
plt.subplot(612)
X_pca_recover.vDiskRead.plot(label='DiskRead_ica')
plt.legend(loc='best')
plt.subplot(613)
X_pca_recover.vDiskWrite.plot(label='DiskWirte_ica')
plt.legend(loc='best')
plt.subplot(614)
X_pca_recover.vNwEgress.plot(label='NwEgress_ica')
plt.legend(loc='best')
plt.subplot(615)
X_pca_recover.vNwIngress.plot(label='NwIngress_ica')
plt.legend(loc='best')
plt.subplot(616)
X_pca_recover.vMemUsage.plot(label='MemUsage_ica')
plt.legend(loc='best')


# In[33]:

from sklearn.metrics import r2_score


# In[84]:

r2_score(X,X_pca_recover,multioutput='variance_weighted')


# >通过PCA降维，只保留4个维度的数据，也能较好的恢复原数据

# In[17]:

#计算T2统计量
X_pca_T2=Series([np.dot(np.dot(np.dot(np.dot(X.ix[i].T,X_pca.components_.T),nlg.inv(np.diag(X_pca.explained_variance_))),X_pca.components_),X.ix[i]) for i in np.arange(len(X))],
         index=X.index)


# In[18]:

#计算SPE统计量
X_pca_SPE=Series(np.sum((X-X_pca_recover)**2,axis=1),index=X.index)


# #### 采用置信度确定阈值

# ##### option1:使用scikit learn 的KDE API估计概率密度

# In[19]:

from sklearn.neighbors import KernelDensity


# In[20]:

X_pca_T2_scikit_kde=KernelDensity().fit(X_pca_T2.reshape(-1,1)) #reshape(-1,1)是API要求,否则视为一个点，概率密度就无从谈起了
X_pca_SPE_scikit_kde=KernelDensity().fit(X_pca_SPE.reshape(-1,1))


# In[21]:

X_pca_T2_sort=X_pca_T2.sort_values()
plt.plot(np.exp(X_pca_T2_scikit_kde.score_samples(X_pca_T2_sort.reshape(-1,1))))


# In[22]:

X_pca_T2_dens_plot=np.linspace(0,50,1000)
plt.plot(np.exp(X_pca_T2_scikit_kde.score_samples(X_pca_T2_dens_plot.reshape(-1,1))))


# In[101]:

from scipy.integrate import quad


# In[106]:

def X_pca_T2_scikit_kde_func(x):
    return np.exp(X_pca_T2_scikit_kde.score_samples(x))

def X_pca_SPE_scikit_kde_func(x):
    return np.exp(X_pca_SPE_scikit_kde.score_samples(x))

def get_threshold_of_scikit_kde(kde,start,step=1,confidence=0.997):
    """get threshold by confidence"""
    i = start
    cumsum = quad(kde,-np.inf, start)[0]
    while True:
        if cumsum >= confidence:
            break
        cumsum = cumsum + quad(kde,i, i+step)[0]
        i = i + step
        
    return i

get_threshold_of_scikit_kde(X_pca_T2_scikit_kde_func,X_pca_T2.min())


# ##### option2:使用scipy 的KDE API估计概率密度

# In[23]:

from scipy import stats


# In[24]:

def my_kde_bandwidth(obj, fac=1./5):
    """We use Scott's Rule, multiplied by a constant factor."""
    return np.power(obj.n, -1./(obj.d+4)) * fac

X_pca_T2_scipy_kde=stats.gaussian_kde(X_pca_T2, bw_method=my_kde_bandwidth)
X_pca_SPE_scipy_kde=stats.gaussian_kde(X_pca_SPE, bw_method=my_kde_bandwidth)


# In[25]:

plt.plot(X_pca_T2_scipy_kde.pdf(X_pca_T2_sort))


# In[26]:

plt.plot(X_pca_T2_scipy_kde.pdf(X_pca_T2_dens_plot))


# In[107]:

def get_threshold_of_scipy_kde(kde,start,step=1,confidence=0.997):
    """get threshold by confidence"""
    i = start
    cumsum = kde.integrate_box_1d(-np.inf, start)
    while True:
        if cumsum >= confidence:
            break
        cumsum = cumsum + kde.integrate_box_1d(i, i+step)
        i = i + step
        
    return i

get_threshold_of_scipy_kde(X_pca_T2_scipy_kde,X_pca_T2.min())


# > 采用scikit-learn与scipy均可通过KDE进行阈值的确定，结果一致。但scipy无论是使用便捷性还是计算效率（ms级）都比scikit-learn（s级）要好，故后续使用scipy来进行计算。

# In[28]:

plt.subplot(211)
plt.plot(X_pca_T2.values,label='PCA-T2')
plt.plot(get_threshold_of_scipy_kde(X_pca_T2_scipy_kde,X_pca_T2.min(),confidence=0.997)*np.ones(len(X_pca_T2)),'r--')
plt.legend(loc='best')

plt.subplot(212)
plt.plot(X_pca_SPE.values,label='PCA-SPE')
plt.plot(get_threshold_of_scipy_kde(X_pca_SPE_scipy_kde,X_pca_SPE.min(),confidence=0.997)*np.ones(len(X_pca_SPE)),'r--')
plt.legend(loc='best')


# #### 检测到的异常时刻

# ##### T2检测

# In[288]:

X_pca_T2_anomaly=X_pca_T2[X_pca_T2>get_threshold_of_scipy_kde(X_pca_T2_scipy_kde,X_pca_T2.min(),confidence=0.997)].index
#10min聚合，注意第一个元素的处理
indice=pd.Series([True]+list(np.diff(X_pca_T2_anomaly)>10))
X_pca_T2_anomaly_plot=Series(np.ones(len(X_pca_T2_anomaly[indice])),index=X_pca_T2_anomaly[indice])
X_pca_T2_anomaly[indice]


# ##### SPE检测

# In[243]:

X_pca_SPE_anomaly=X_pca_SPE[X_pca_SPE>get_threshold_of_scipy_kde(X_pca_SPE_scipy_kde,X_pca_SPE.min(),confidence=0.997)].index
indice=pd.Series([True]+list(np.diff(X_pca_SPE_anomaly)>10))
X_pca_SPE_anomaly_plot=Series(np.ones(len(X_pca_SPE_anomaly[indice])),index=X_pca_SPE_anomaly[indice])
X_pca_SPE_anomaly[indice]


# In[286]:

anomaly_plot=pd.concat([X_pca_T2_anomaly_plot,X_pca_SPE_anomaly_plot],axis=1)
anomaly_plot.columns=['T2','SPE']
anomaly_plot.plot(kind='bar')


# > 将SPE与T2同时检测到异常的时刻确定为异常时刻，考虑到采样和插值误差，可以将SPE 10min范围内出现的T2视为同时出现

# ### ICA

# In[30]:

from sklearn.decomposition import FastICA


# #### 使用混合矩阵能量占比确定独立成分

# In[31]:

def get_ica_components(X,contribution=0.85):
    X_ica=FastICA(n_components=len(X.columns)).fit(X)    
    
    L2=Series(np.sum(X_ica.mixing_**2,axis=0))
    L2.sort_values(ascending=False,inplace=True)
    
    X_S=DataFrame(X_ica.transform(X))
    X_ica_mixing_=DataFrame(X_ica.mixing_)

    L2.drop(L2.index[L2.cumsum()/L2.sum()>=contribution][1:],inplace=True)

    return X_S.reindex(columns=L2.index).values,X_ica_mixing_.reindex(columns=L2.index).values,X_ica.mean_,len(L2)


# In[32]:

X_S_,X_ica_mixing_,X_ica_mean_,n=get_ica_components(X)
print(X.shape,X_S_.shape,X_ica_mixing_.shape,n)


# #### 使用R2统计量确定独立成分个数

# In[34]:

def get_ica_n_components(X,r2_score_value=0.85):
    for i in np.arange(len(X.columns)-1):
        X_ica=FastICA(n_components=len(X.columns)-i-1).fit(X)
        X_S=X_ica.transform(X)
        if r2_score(X, np.dot(X_S,X_ica.mixing_.T)+X_ica.mean_,multioutput='variance_weighted') < r2_score_value:
            break;
    return len(X.columns)-i

n=get_ica_n_components(X)
n


# In[35]:

X_ica=FastICA(n_components=n).fit(X)


# In[36]:

X_S_=X_ica.transform(X)


# In[37]:

#各种算法的变量统一，便于后续计算
X_ica_mixing_=X_ica.mixing_
X_ica_mean_=X_ica.mean_


# In[38]:

print(X.shape,X_S_.shape,X_ica_mixing_.shape,X_ica.components_.shape)


# In[39]:

X_ica_recover=DataFrame(np.dot(X_S_,X_ica_mixing_.T)+X_ica_mean_,index=X.index,columns=X.columns)


# In[40]:

plt.subplot(611)
X_ica_recover.vCpuUsage.plot(label='vCpuUsage_ica')
plt.legend(loc='best')
plt.subplot(612)
X_ica_recover.vDiskRead.plot(label='DiskRead_ica')
plt.legend(loc='best')
plt.subplot(613)
X_ica_recover.vDiskWrite.plot(label='DiskWirte_ica')
plt.legend(loc='best')
plt.subplot(614)
X_ica_recover.vNwEgress.plot(label='NwEgress_ica')
plt.legend(loc='best')
plt.subplot(615)
X_ica_recover.vNwIngress.plot(label='NwIngress_ica')
plt.legend(loc='best')
plt.subplot(616)
X_ica_recover.vMemUsage.plot(label='MemUsage_ica')
plt.legend(loc='best')


# In[41]:

#计算T2统计量
X_ica_T2=Series(np.sum(X_S_**2,axis=1),index=X.index)


# In[42]:

#计算SPE统计量
X_ica_SPE=Series(np.sum((X-X_ica_recover)**2,axis=1),index=X.index)


# In[43]:

X_ica_T2_scipy_kde=stats.gaussian_kde(X_ica_T2, bw_method=my_kde_bandwidth)
X_ica_SPE_scipy_kde=stats.gaussian_kde(X_ica_SPE, bw_method=my_kde_bandwidth)


# In[44]:

plt.subplot(211)
plt.plot(X_ica_T2.values,label='ICA-T2')
plt.plot(get_threshold_of_scipy_kde(X_ica_T2_scipy_kde,X_ica_T2.min(),step=0.01,confidence=0.997)*np.ones(len(X_ica_T2)),'r--')
plt.legend(loc='best')

plt.subplot(212)
plt.plot(X_ica_SPE.values,label='ICA-SPE')
plt.plot(get_threshold_of_scipy_kde(X_ica_SPE_scipy_kde,X_ica_SPE.min(),confidence=0.997)*np.ones(len(X_ica_SPE)),'r--')
plt.legend(loc='best')


# #### 检测到的异常时刻

# ##### T2检测

# In[289]:

X_ica_T2_anomaly=X_ica_T2[X_ica_T2>get_threshold_of_scipy_kde(X_ica_T2_scipy_kde,X_ica_T2.min(),step=0.01,confidence=0.997)].index
#10min聚合，注意第一个元素的处理
indice=pd.Series([True]+list(np.diff(X_ica_T2_anomaly)>10))
X_ica_T2_anomaly_plot=Series(np.ones(len(X_ica_T2_anomaly[indice])),index=X_ica_T2_anomaly[indice])
X_ica_T2_anomaly[indice]


# ##### SPE检测

# In[290]:

X_ica_SPE_anomaly=X_ica_SPE[X_ica_SPE>get_threshold_of_scipy_kde(X_ica_SPE_scipy_kde,X_ica_SPE.min(),confidence=0.997)].index
indice=pd.Series([True]+list(np.diff(X_ica_SPE_anomaly)>10))
X_ica_SPE_anomaly_plot=Series(np.ones(len(X_ica_SPE_anomaly[indice])),index=X_ica_SPE_anomaly[indice])
X_ica_SPE_anomaly[indice]


# In[291]:

anomaly_plot=pd.concat([X_ica_T2_anomaly_plot,X_ica_SPE_anomaly_plot],axis=1)
anomaly_plot.columns=['T2','SPE']
anomaly_plot.plot(kind='bar')


# > ica检测效果明显优于pca，理论上应该是原始数据分布更趋向于非高斯，比较适合ica算法的假设

# ### LLE

# In[46]:

from sklearn import manifold


# ##### 考察是否具有流行特征

# In[296]:

#降到3维观察
X_lle_embed=manifold.LocallyLinearEmbedding(n_neighbors =10, n_components=3).fit_transform(X)


# In[297]:

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig, elev=30, azim=-20)
ax.scatter(X_lle_embed[:,0], X_lle_embed[:,1], X_lle_embed[:,2], marker='o', cmap=plt.cm.rainbow)


# > 从图示看，此3D图形并没有复杂的流行结构，感觉采用流行算法的必要性不高，体现不出LLE不受数据分布约束的优势。而且由于LLE的k，d参数难有较好的手段确定，且计算复杂度也比较高，因此不推荐使用。这里出于研究目的，仍给出LLE的算法参考。

# In[298]:

X_lle_embed=manifold.LocallyLinearEmbedding(n_neighbors =10, n_components=5).fit_transform(X)


# In[299]:

X_lle_embed_mean=X_lle_embed.mean(axis=0)
X_lle_embed_mean


# In[300]:

X_lle_vi=nlg.inv(np.cov(X_lle_embed.T))
X_lle_vi


# In[301]:

#计算的是到均值的马氏距离
X_lle_T2=Series([np.dot(np.dot(X_lle_embed[i]-X_lle_embed_mean,X_lle_vi),X_lle_embed[i]-X_lle_embed_mean)
                 for i in np.arange(len(X_lle_embed))],index=X.index)


# In[302]:

X_lle_T2.describe()


# In[303]:

X_lle_T2_scipy_kde=stats.gaussian_kde(X_lle_T2, bw_method=my_kde_bandwidth)


# In[304]:

plt.plot(X_lle_T2.values,label='LLE-T2')
plt.plot(get_threshold_of_scipy_kde(X_lle_T2_scipy_kde,X_lle_T2.min(),confidence=0.997)*np.ones(len(X_lle_T2)),'r--')
plt.legend(loc='best')


# #### 检测到的异常时刻

# In[305]:

X_lle_T2_anomaly=X_lle_T2[X_lle_T2>get_threshold_of_scipy_kde(X_lle_T2_scipy_kde,X_lle_T2.min(),confidence=0.997)].index
#10min聚合，注意第一个元素的处理
indice=pd.Series([True]+list(np.diff(X_lle_T2_anomaly)>10))
X_lle_T2_anomaly[indice]


# > T2误检时刻较多。由于LLE算法是非线性算法，无法转换到原始数据空间，因此不能计算SPE。

# ### 不降维，直接采用马氏距离平方进行度量

# > 由于特征较少，出于研究目的，本小节给出不降维的异常检测结果用于对比

# In[56]:

from scipy import spatial


# In[57]:

#计算每个样本到均值的马氏距离
X_vi=nlg.inv(np.cov(X.T))
X_mahalanobis_dist=[spatial.distance.pdist(np.array([X.ix[i],X_mean]),metric='mahalanobis',VI=X_vi)[0] for i in np.arange(len(X))]


# In[58]:

X_mahalanobis_dist2=Series(np.square(X_mahalanobis_dist),index=X.index)


# In[59]:

X_mahalanobis_dist2_scipy_kde=stats.gaussian_kde(X_mahalanobis_dist2, bw_method=my_kde_bandwidth)


# In[306]:

plt.plot(X_mahalanobis_dist2.values,label='mahalanobis-distance2')
plt.plot(get_threshold_of_scipy_kde(X_mahalanobis_dist2_scipy_kde,X_mahalanobis_dist2.min(),confidence=0.997)
         *np.ones(len(X_mahalanobis_dist2)),'r--')
plt.legend(loc='best')


# #### 检测到的异常时刻

# In[295]:

X_mahalanobis_dist2_anomaly=X_mahalanobis_dist2[X_mahalanobis_dist2>get_threshold_of_scipy_kde(X_mahalanobis_dist2_scipy_kde,X_mahalanobis_dist2.min(),confidence=0.997)].index
#10min聚合，注意第一个元素的处理
indice=pd.Series([True]+list(np.diff(X_mahalanobis_dist2_anomaly)>10))
X_mahalanobis_dist2_anomaly[indice]

