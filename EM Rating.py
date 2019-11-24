import pandas as pd
import pathlib
import numpy as np
locationUsersFile = pathlib.Path(r'data/purchase_play.csv')
data = pd.read_csv(locationUsersFile,header=0)

def Generate_EM(game,rating,print_vals=TRUE):
  gdata=data[game,data.hrs>2]
  gdata['loghrs']=log(gdata['hrs'])
  mui=np.linespace(min(gdata['loghrs']),max(gdata['loghrs']),length=rating)
  EM=normalmixEm(gdata['loghrs'].mu=mui,sigma=rep(1,nclass))


def get_pdf(sample, mu, sigma):
    res = stats.multivariate_normal(mu, sigma).pdf(sample)
    return res


def get_log(data, k, mu, sigma, gama):
    res = 0.0
    for i in range(len(data)):
        cur = 0.0
        for j in range(len(k)):
            cur += gama[j][i] * get_pdf(data[i], mu[j], sigma[j])
        res += math.log(cur)
    return res

def EM(data,k,mu,sigma,steps=1000):
    num_gau = len(k)  
    num_data = data.shape[0]  
    gama = np.zeros((num_gau, num_data)) 
    record = []  
    for step in range(steps):
        #gama matrix
        for i in range(num_gau):
            for j in range(num_data):
                gama[i][j] = k[i] * get_pdf(data[j], mu[i], sigma[i]) / \
                             sum([k[t] * get_pdf(data[j], mu[t], sigma[t]) for t in range(num_gau)])
        cur_game = get_log(data, k, mu, sigma, gama)  # calculate log
        record.append(cur_game)
        # update mu
        for i in range(num_gau):
            mu[i] = np.dot(gama[i], data) / np.sum(gama[i])
        # update sigma
        for i in range(num_gau):
            cov = [np.dot((data[t] - mu[i]).reshape(-1, 1), (data[t] - mu[i]).reshape(1, -1)) for t in range(num_data)]
            cov_sum = np.zeros((2, 2))
            for j in range(num_data):
                cov_sum += gama[i][j] * cov[j]
            sigma[i] = cov_sum / np.sum(gama[i])
        # update k
        for i in range(num_gau):
            k[i] = np.sum(gama[i]) / num_data
        print('step: {}\t game:{}'.format(step + 1, cur_game))
    return k, mu, sigma, gama, record

def main():
    
