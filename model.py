import tensorflow as tf
from amplitude import AllAmplitude
import time 
import functools

class Model:
  def __init__(self,res,w_bkg = 0):
    self.Amp = AllAmplitude(res)
    self.w_bkg = w_bkg
    
  def nll(self,data,bg,mcdata):
    ln_data = tf.reduce_sum(tf.math.log(self.Amp(data)))
    ln_bg = tf.reduce_sum(tf.math.log(self.Amp(bg)))
    int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    n_mc = mcdata[0].shape[0]
    return -(ln_data - self.w_bkg * ln_bg - (n_data - self.w_bkg*n_bg) * int_mc)
  
  def nll_gradient(self,data,bg,mcdata,batch):
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    n_mc = mcdata[0].shape[0]
    N = len(data)
    data_warp = [tf.concat([data[i],bg[i]],0) for i in range(N)]
    data_weight = tf.concat([tf.ones(shape=(n_data,),dtype="float32"),-tf.ones(shape=(n_bg,),dtype="float32")*self.w_bkg],0)
    sum_w = n_data - self.w_bkg * n_bg
    nll,g = self.sum_gradient(data_warp,data_weight,batch,func=tf.math.log)
    s,g2 = self.sum_gradient(mcdata,1/n_mc,batch)
    for i in range(len(g)):
      g[i] = -g[i] + sum_w * g2[i]/s
    return -nll+sum_w*tf.math.log(s),g
  
  def sum_gradient(self,data,weight=1.0,batch=1536,func = lambda x:x):
    data_i = []
    N = len(data)
    n_data = data[0].shape[0]
    n_split = (n_data + batch -1) // batch
    for i in range(n_split):
      data_i.append([
        data[j][i*batch:min(i*batch+batch,n_data)] for j in range(N)
      ])
    data_wi = []
    if isinstance(weight,float):
      weight = tf.ones(n_data)*weight
    for i in range(n_split):
      data_wi.append(weight[i*batch:min(i*batch+batch,n_data)])
    g = None
    nll = 0.0
    n_variables = len(self.Amp.trainable_variables)
    for i in range(n_split):
      #print(i,min(i*batch+batch,n_data))
      with tf.GradientTape() as tape:
        amp2s = self.Amp(data_i[i])
        l_a = func(amp2s)
        p_nll = tf.reduce_sum(data_wi[i] * l_a)
      nll += p_nll
      a = tape.gradient(p_nll,self.Amp.trainable_variables)
      if g is None:
        g = a
      else :
        for j in range(n_variables):
          g[j] += a[j]
    return nll,g

param_list = [
  "m_BC","m_BD","m_CD",
  "cosTheta_BC","cosTheta_B_BC",
  "phi_BC", "phi_B_BC",
  "cosTheta_BD","cosTheta_D_BD",
  "phi_D_BD",
  "cosTheta_CD","cosTheta_C_CD",
  "phi_CD","phi_C_CD",
  "cosTheta1","cosTheta2",
  "phi1","phi2"
]

config_list = {"D2_2460"
    :{
        "m0":2.4607,
        "m_min":2.4603,
        "m_max":2.4611,
        "g0":0.0475,
        "g_min":0.0464,
        "g_max":0.0486,
        "J":2,
        "Par":1,
        "Chain":21
    },
    "D2_2460p"
    :{
        "m0":2.4654,
        "m_min":2.4644,
        "m_max":2.4667,
        "g0":0.0467,
        "g_min":0.0455,
        "g_max":0.0479,
        "J":2,
        "Par":1,
        "Chain":121
    },
    "D1_2430"
    :{
        "m0":2.427,
        "m_min":2.387,
        "m_max":2.467,
        "g0":0.284,
        "g_min":0.274,
        "g_max":0.514,
        "J":1,
        "Par":1,
        "Chain":12
    },
    "D1_2430p"
    :{
        "m0":2.427,
        "m_min":2.387,
        "m_max":2.467,
        "g0":0.384,
        "g_min":0.274,
        "g_max":0.514,
        "J":1,
        "Par":1,
        "Chain":112
    },
    "D1_2420"
    :{
        "m0":2.4208,
        "m_min":2.4203,
        "m_max":2.4213,
        "g0":0.0317,
        "g_min":0.0292,
        "g_max":0.0342,
        "J":1,
        "Par":1,
        "Chain":11
    },
    "D1_2420p"
    :{
        "m0":2.4232,
        "m_min":2.4208,
        "m_max":2.4256,
        "g0":0.025,
        "g_min":0.019,
        "g_max":0.021,
        "J":1,
        "Par":1,
        "Chain":111
    },
    "Zc_4025"
    :{
        "m0":4.0263,
        "g0":0.0248,
        "J":1,
        "Par":1,
        "Chain":-1
    },
    "Zc_4160"
    :{
        "m0":4.1628,
        "g0":0.0701,
        "J":1,
        "Par":1,
        "Chain":-2
    }
}

def train_one_step(model, optimizer, data, bg,mc,batch=16384):
  nll,grads = model.nll_gradient(data,bg,mc,batch)
  print(grads)
  print(nll)
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  #with tf.GradientTape() as tape:
    #nll = model.nll(data,bg,mc)
  #g = tape.gradient(nll,model.Amp.trainable_variables)
  #print(nll,g)
  return nll

class fcn(object):
  """
  provide FCN function and gradient for minuit
  """
  def __init__(self,model,data,bg,mc,batch=16384):
    self.model = model
    self.data = data
    self.bg = bg
    self.mc = mc
    self.batch = batch
    self.grads = []
    self.x = None
    self.nll = 0.0
    w_bkg = model.w_bkg
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    self.alpha = (n_data - w_bkg * n_bg)/(n_data + w_bkg**2 * n_bg)
  def __call__(self,*x):
    now = time.time()
    if (not self.x is None) and self.x == x:
      return self.nll
    self.x = x
    train_vars = self.model.Amp.trainable_variables
    n_var = len(train_vars)
    for i in range(n_var):
      train_vars[i].assign(x[i])
    nll,g = self.model.nll_gradient(self.data,self.bg,self.mc,self.batch)
    self.grads = [ self.alpha * i.numpy() for i in g]
    print("nll:",self.alpha * nll," time :",time.time() - now)
    return self.alpha * nll
  
  @functools.lru_cache()
  def grad(self,*x):
    if (not self.x is None) and self.x == x:
      return self.grads
    self(*x)
    return self.grads
  
def set_gpu_mem_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

def main():
  import json,time
  set_gpu_mem_growth()
  a = Model(config_list,0.8)
  data = []
  bg = []
  mcdata = []
  with open("./data/PHSP_ang.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      mcdata.append(tmp_data)
  with open("./data/data_ang.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      data.append(tmp_data)
  with open("./data/bg_ang.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      bg.append(tmp_data)
  #print(data,bg,mcdata)
  import iminuit 
  f = fcn(a,data,bg,mcdata,27648)# 1356*18
  args = {}
  args_name = []
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    args_name.append(i.name)
    args["error_"+i.name] = 0.1
  m = iminuit.Minuit(f,forced_parameters=args_name,errordef = 0.5,print_level=2,grad=f.grad,**args)
  now = time.time()
  with tf.device('/device:GPU:0'):
    m.migrad()
  print(time.time() - now)
  m.get_param_states()
  exit()
  data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
  #data_set = data_set.shuffle(10000).batch(800)
  data_set_it = iter(data_set)
  bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
  #bg_set = bg_set.shuffle(10000).batch(340)
  bg_set_it = iter(bg_set)
  mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
  #mc_set = mc_set.shuffle(10000).batch(2520)
  mc_set_it = iter(mc_set)
  now = time.time()
  with tf.device('/device:GPU:0'):
    print(a.nll(data,bg,mcdata))#.collect_params())
  optimizer = tf.keras.optimizers.Adadelta(1.0)
  for i in range(100):
    #try :
      #data_i = data_set_it.get_next()
      #bg_i = bg_set_it.get_next()
      #mcdata_i = mc_set_it.get_next()
    train_one_step(a,optimizer,data,bg,mcdata)
    #except:
      #data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
      ##data_set = data_set.shuffle(10000).batch(800)
      #data_set_it = iter(data_set)
      #bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
      ##bg_set = bg_set.shuffle(10000).batch(340)
      #bg_set_it = iter(bg_set)
      #mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
      ##mc_set = mc_set.shuffle(10000).batch(2520)
      #mc_set_it = iter(mc_set)
  print(time.time()-now)
  #now = time.time()
  #with tf.device('/device:CPU:0'):
    #print(a(x))#.collect_params())
  #print(time.time()-now)
  with tf.device('/device:GPU:0'):
    print(a.nll(data,bg,mcdata))#.collect_params())
  print(a.Amp.trainable_variables)
  
if __name__=="__main__":
  main()
