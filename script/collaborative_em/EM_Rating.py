import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import re
from plotnine import *
import scipy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


locationUsersFile=pathlib.Path(r'D:/Game-Recommendation-System/data/raw_data/steam_users_purchase_play.csv')
steam_clean = pd.read_csv(locationUsersFile, header=1, names=['user', 'game', 'hrs', 'purchase','play'])

locationUsersFile_train=pathlib.Path(r'D:/Game-Recommendation-System/data/model_data/steam_user_train.csv')
steam_traind = pd.read_csv(locationUsersFile_train, header=1, names=['user', 'game', 'hrs', 'purchase','play'])

locationUsersFile_test=pathlib.Path(r'D:/Game-Recommendation-System/data/model_data/steam_user_test.csv')
steam_test = pd.read_csv(locationUsersFile_test, header=1, names=['user', 'game', 'hrs', 'purchase','play'])

game_freq = steam_traind.groupby(by='game').agg({'user': 'count', 'hrs': 'sum'}).reset_index()
#game_freq = steam_clean.groupby(by='game').agg({'user': 'count', 'hrs': 'sum'}).reset_index()
top20 = game_freq.sort_values(by='user',ascending=False)[:20].reset_index()
#print(top20)
steam_traind['user']=steam_traind['user'].astype(int)
steam_clean['user']=steam_clean['user'].astype(int)
steam_test['user']=steam_test['user'].astype(int)

# Cleaning up the game columns. It doesn't like some of the special characters
steam_traind['game1'] = steam_traind['game'].apply(lambda x: re.sub('[^a-zA-Z0-9]', '', x))
steam_clean['game1'] = steam_clean['game'].apply(lambda x: re.sub('[^a-zA-Z0-9]', '', x))
#steam_clean.head()

#EM Algorithm based on raw data
def game_hrs_density(GAME, nclass, print_vals=True):
    #Ignore the game hrs less than 2 hrs
    #game_data = steam_clean[(steam_clean['game1'] == GAME) & (steam_clean['hrs'] > 2)]
    game_data = steam_clean[(steam_clean['game1'] == GAME)&(steam_clean['hrs']>2)]
    #Log hrs
    game_data['loghrs'] = np.log(steam_clean['hrs'])
    #Calculate the mu,sigma to process Gaussian function
    mu_init = np.linspace(min(game_data['loghrs']), max(game_data['loghrs']), nclass).reshape(-1, 1)
    sigma_init = np.array([1] * nclass).reshape(-1, 1, 1)
    gaussian = GaussianMixture(n_components=nclass, means_init=mu_init, precisions_init=sigma_init).fit(game_data['loghrs'].values.reshape([-1, 1]))
    #print Gaussian Lambda Mean Sigma
    if print_vals:
        print(' lambda: {}\n mean: {}\n sigma: {}\n'.format(gaussian.weights_, gaussian.means_, gaussian.covariances_))
    #Random Generate
    x = np.linspace(min(game_data['loghrs']), max(game_data['loghrs']), 1000)
    #Plot
    dens = pd.DataFrame({'x': x})
    for i in range(nclass):
        dens['y{}'.format(i+1)] = gaussian.weights_[i]* scipy.stats.norm(gaussian.means_[i][0], gaussian.covariances_[i][0][0]).pdf(x)
    dens = dens.melt('x', value_name='gaussian')
    # Building data frame for plotting
    game_plt = ggplot(aes(x='loghrs', y='stat(density)'), game_data) + geom_histogram(bins=25, colour = "black", alpha = 0.7, size = 0.1) + \
               geom_area(dens, aes(x='x', y='gaussian', fill = 'variable'), alpha = 0.5, position = position_dodge(width=0.2)) + geom_density()+ \
               ggtitle(GAME)
    return game_plt
#Print one example
a = game_hrs_density('Fallout4', 5, True)
print(a)


# Create user item matrix
np.random.seed(910)
# Delete unnecessary characters
game_freq['game1'] = game_freq['game'].apply(lambda x: re.sub('[^a-zA-Z0-9]', '', x))
# Only Consider the games have more than 50 users
game_users = game_freq[game_freq['user'] > 50]

#For whole dataset
steam_clean_pos = steam_clean[steam_clean['hrs'] > 2]
steam_clean_pos_idx = steam_clean_pos['game1'].apply(lambda x: x in game_users['game1'].values)
steam_clean_pos = steam_clean_pos[steam_clean_pos_idx]
steam_clean_pos['loghrs'] = np.log(steam_clean_pos['hrs'])


# make matrix
games = pd.DataFrame({'game1': sorted(steam_clean_pos['game1'].unique()), 'game_id': range(len(steam_clean_pos['game1'].unique()))})
users = pd.DataFrame({'user': sorted(steam_clean_pos['user'].unique()), 'user_id': range(len(steam_clean_pos['user'].unique()))})
steam_clean_pos = pd.merge(steam_clean_pos, games, on=['game1'])
steam_clean_pos = pd.merge(steam_clean_pos, users, on=['user'])

ui_mat = np.zeros([len(users), len(games)])
for i in range(steam_clean_pos.shape[0]):
    line = steam_clean_pos.iloc[i]
    ui_mat[line['user_id'], line['game_id']] = line['loghrs']

#test dataset user
users_test = pd.DataFrame({'user': sorted(steam_test['user'].unique()), 'user_id': range(len(steam_test['user'].unique()))})
#print(users_test)

# For train dataset
# Only consider the games hrs more than 2 hrs
steam_train = steam_traind[steam_traind['hrs'] > 2]
#print(steam_train)
#Not consider the games that users less than 50
steam_train_idx = steam_train['game1'].apply(lambda x: x in game_users['game1'].values)
steam_train = steam_train[steam_train_idx]
steam_train['loghrs'] = np.log(steam_train['hrs'])
# Make Matrix
# List the games in train dataset use for recommend
games_train = pd.DataFrame({'game1': sorted(steam_train['game1'].unique()), 'game_id': range(len(steam_train['game1'].unique()))})
# List the users in train dataset use for recommend
users_train = pd.DataFrame({'user': sorted(steam_train['user'].unique()), 'user_id': range(len(steam_train['user'].unique()))})
#Merge the games and users to one data frame
steam_train = pd.merge(steam_train, games_train, on=['game1'])
steam_train = pd.merge(steam_train, users_train, on=['user'])

# Create training set
test=steam_train
ui_train = ui_mat
for i in range(test.shape[0]):
    line = test.iloc[i]
    ui_train[line['user_id'], line['game_id']] = 0
print("Dimensions of training user-item matrix:", ui_train.shape)

# Root Mean Squared error function, Evaluation metric for SVD
def rmse(pred, test, data_frame=False):
    test_pred = np.array([np.nan] * len(test))
    for i in range(len(test)):
        line = test.iloc[i]
        test_pred[i] = pred[line['user_id'], line['game_id']]
    if data_frame:
        return pd.DataFrame({'test_pred': test_pred, 'loghrs': test['loghrs']})
    return np.sqrt(1/(len(test)-1)*np.sum((test_pred - test['loghrs']) ** 2))




# Basic svd
Y = pd.DataFrame(ui_train).copy()

# Impute the missing observations with a mean value
means = np.mean(Y)
for i, col in enumerate(Y.columns):
    Y[col] = Y[col].apply(lambda x: means[i] if x == 0 else x)
U, D, V = np.linalg.svd(Y)
p_df = pd.DataFrame({'x': range(1, len(D)+1), 'y': D/np.sum(D)})

'''
ggplot(p_df, aes(x='x', y='y')) + \
geom_line() + \
labs(x = "Leading component", y = "")
'''
#Set the latent factor as 60
lc = 60
pred = np.dot(np.dot(U[:, :lc], np.diag(D[:lc])), V[:lc, :])
#Calculate rmse
#print(rmse(pred, test))
#rmse(pred, test, True).head()

#SVD via gradient descent
#Set the latent factor as 60
leading_components=60

# Setting matricies
Y = pd.DataFrame(ui_train)
I = Y.copy()

for col in I.columns:
    I[col] = I[col].apply(lambda x: 1 if x > 0 else 0)
U = np.random.normal(0, 0.01, [I.shape[0], leading_components])
V = np.random.normal(0, 0.01, [I.shape[1], leading_components])
#Squared error
def f(U, V):
    return np.sum(I.values*(np.dot(U, V.T)-Y.values)**2)
def dfu(U):
    return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)), V)
def dfv(V):
    return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)).T, U)

#Gradient descent
N = 200
alpha = 0.001
pred = np.round(np.dot(U, V.T), decimals=2)
fobj = [f(U, V)]
rmsej = [rmse(pred, test)]
start = time.time()
#process iteratively until we get to the bottom
for i in tqdm(range(N)):
    U = U - alpha*dfu(U)
    V = V - alpha*dfv(V)
    fobj.append(f(U, V))
    pred = np.round(np.dot(U, V.T), 2)
    rmsej.append(rmse(pred, test))

#print('Time difference of {} mins'.format((time.time() - start) / 60))
#fojb predicted values
fojb = np.array(fobj)
#rmsej actual observed values
rmsej = np.array(rmsej)
path1 = pd.DataFrame({'itr': range(1, N+2), 'fobj': fobj, 'fobjp': fobj/max(fobj), 'rmse': rmsej, 'rmsep': rmsej/max(rmsej)})
path1gg = pd.melt(path1[["itr", "fobjp", "rmsep"]], id_vars=['itr'])
print(path1.tail(1))

print(ggplot(path1gg, aes('itr', 'value', color = 'variable')) + geom_line())

# Create a rating based on time played after gradient descent
def game_hrs_density_p(pred, GAME=None, nclass=1, print_vals=True):
    game_dict = dict(games.values)
    t_GAME = GAME
    if not GAME:
        GAME = np.random.randint(0, games.shape[0])
    else:
        GAME = game_dict[GAME]
    game_data = pd.Series(pred[:, GAME])
    game_data = game_data[game_data > 0]

    # EM algorithm
    mu_init = np.linspace(min(game_data), max(game_data), nclass).reshape(-1, 1)
    sigma_init = np.array([1] * nclass).reshape(-1, 1, 1)
    gaussian = GaussianMixture(n_components=nclass, means_init=mu_init, precisions_init=sigma_init).fit(game_data.values.reshape([-1, 1]))
    if print_vals:
        print(' lambda: {}\n mean: {}\n sigma: {}\n'.format(gaussian.weights_, gaussian.means_, gaussian.covariances_))
    # building data frame for plotting
    x = np.linspace(min(game_data), max(game_data), 1000)
    dens = pd.DataFrame({'x': x})
    for i in range(nclass):
        dens['y{}'.format(i+1)] = gaussian.weights_[i]* scipy.stats.norm(gaussian.means_[i][0], gaussian.covariances_[i][0][0]).pdf(x)
    dens = dens.melt('x', value_name='gaussian')
    game_data = pd.DataFrame(game_data, columns=['game_daat'])
    game_plt = ggplot(aes(x='game_data', y='stat(density)'), game_data) + geom_histogram(bins=45, colour = "black", alpha = 0.7, size = 0.1) + \
               geom_area(dens, aes(x='x', y='gaussian', fill = 'variable'), alpha = 0.5, position = position_dodge(width=0.2)) + geom_density()+ \
               ggtitle(t_GAME)
    return game_plt

a = game_hrs_density_p(pred, "Fallout4", 5)
print(a)

# Export recommend games
user_dict = dict(users.values)
game_dict = {games.iloc[i, 0]: games.iloc[i, 1] for i in range(games.shape[0])}
I_pred = np.zeros_like(I)
for i in range(steam_clean.shape[0]):
    line = steam_clean.iloc[i]
    if line['user'] in user_dict and line['game1'] in game_dict:
        I_pred[user_dict[line['user']], game_dict[line['game1']]] = 1

reverse_game_dict = {games.iloc[i, 1]: games.iloc[i, 0] for i in range(games.shape[0])}
pred_percentile = pd.DataFrame(pred)
for col in pred_percentile.columns:
    pred_percentile[col] = pred_percentile[col].rank(pct=True)
pred_percentile = pred_percentile.values

def top(n, user, print_value=True):
    #Not consider the games has been purchsed
    not_purchased = (I - 1) % 2
    t_user = user
    user = user_dict[user]
    top_games = (pred_percentile*not_purchased).iloc[user]
    top_games = list(top_games.sort_values(ascending=False)[:20].index)
    #For test
    if print_value:
        print('top {} recommended games for user {}: '.format(n, t_user))
        for i in range(n):
            print(i, ")", reverse_game_dict[top_games[i]])
    else:
        result = [t_user]
        for i in range(n):
            result.append(reverse_game_dict[top_games[i]])
        return result
#top(20, 5250)

top_N = 20
result = []
#print(users_test['user'])
#print(users_train['user'])
users_merge=pd.merge(users_test,users_train,on='user',how='inner')
#print(users_merge)
for idx, user in tqdm(enumerate(users_merge['user'].values)):
    result.append(top(top_N, user, False))

users_not=users_test[~users_test['user'].isin(users_merge['user'])]
#print(users_not)
for user in users_not['user']:
    empty=[user]
    for i in range(20):
        empty.append(0)
    result.append(empty)
df = pd.DataFrame(result)
columns = ['user_id'] + ['{}'.format(i+1) for i in range(top_N)]
df.columns = columns
df.to_csv('D:/Game-Recommendation-System/data/output_data/Collaborative_EM_output.csv', index=None)



