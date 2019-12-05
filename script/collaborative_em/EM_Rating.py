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
from sklearn.model_selection import train_test_split


locationUsersFile=pathlib.Path(r'D:/Game-Recommendation-System/data/raw_data/steam_users_purchase_play.csv')
steam_clean = pd.read_csv(locationUsersFile, header=1, names=['user', 'game', 'hrs', 'purchase','play'])

game_freq = steam_clean.groupby(by='game').agg({'user': 'count', 'hrs': 'sum'}).reset_index()
top20 = game_freq.sort_values(by='user',ascending=False)[:20].reset_index()
#print(top20)

# Cleaning up the game columns. It doesn't like some of the special characters
steam_clean['game1'] = steam_clean['game'].apply(lambda x: re.sub('[^a-zA-Z0-9]', '', x))
#steam_clean.head()

#First EM Algorithm
def game_hrs_density(GAME, nclass, print_vals=True):
    game_data = steam_clean[(steam_clean['game1'] == GAME) & (steam_clean['hrs'] > 2)]
    game_data['loghrs'] = np.log(steam_clean['hrs'])
    mu_init = np.linspace(min(game_data['loghrs']), max(game_data['loghrs']), nclass).reshape(-1, 1)
    sigma_init = np.array([1] * nclass).reshape(-1, 1, 1)
    gaussian = GaussianMixture(n_components=nclass, means_init=mu_init, precisions_init=sigma_init).fit(game_data['loghrs'].values.reshape([-1, 1]))
    if print_vals:
        print(' lambda: {}\n mean: {}\n sigma: {}\n'.format(gaussian.weights_, gaussian.means_, gaussian.covariances_))
    # building data frame for plotting
    x = np.linspace(min(game_data['loghrs']), max(game_data['loghrs']), 1000)
    dens = pd.DataFrame({'x': x})
    for i in range(nclass):
        dens['y{}'.format(i+1)] = gaussian.weights_[i]* scipy.stats.norm(gaussian.means_[i][0], gaussian.covariances_[i][0][0]).pdf(x)
    dens = dens.melt('x', value_name='gaussian')
    game_plt = ggplot(aes(x='loghrs', y='stat(density)'), game_data) + geom_histogram(bins=25, colour = "black", alpha = 0.7, size = 0.1) + \
               geom_area(dens, aes(x='x', y='gaussian', fill = 'variable'), alpha = 0.5, position = position_dodge(width=0.2)) + geom_density()+ \
               ggtitle(GAME)
    return game_plt

a = game_hrs_density('Fallout4', 5, True)


# Create user item matrix
np.random.seed(910)
game_freq['game1'] = game_freq['game'].apply(lambda x: re.sub('[^a-zA-Z0-9]', '', x))
game_users = game_freq[game_freq['user'] > 50]
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

# create training set i.e. suppress a tenth of the actual ratings
train, test = train_test_split(steam_clean_pos, test_size=0.1)
ui_train = ui_mat
for i in range(test.shape[0]):
    line = test.iloc[i]
    ui_train[line['user_id'], line['game_id']] = 0

# root mean squared error function
def rmse(pred, test, data_frame=False):
    test_pred = np.array([np.nan] * len(test))
    for i in range(len(test)):
        line = test.iloc[i]
        test_pred[i] = pred[line['user_id'], line['game_id']]
    if data_frame:
        return pd.DataFrame({'test_pred': test_pred, 'loghrs': test['loghrs']})
    return np.sqrt(1/(len(test)-1)*np.sum((test_pred - test['loghrs']) ** 2))
print("Dimensions of training user-item matrix:", ui_train.shape)



# Basic svd
Y = pd.DataFrame(ui_train).copy()

# mean impute
means = np.mean(Y)
for i, col in enumerate(Y.columns):
    Y[col] = Y[col].apply(lambda x: means[i] if x == 0 else x)
U, D, V = np.linalg.svd(Y)

p_df = pd.DataFrame({'x': range(1, len(D)+1), 'y': D/np.sum(D)})
ggplot(p_df, aes(x='x', y='y')) + \
geom_line() + \
labs(x = "Leading component", y = "")

lc = 60
pred = np.dot(np.dot(U[:, :lc], np.diag(D[:lc])), V[:lc, :])

#print(rmse(pred, test))
rmse(pred, test, True).head()

# SVD via gradient descent
# Setting matricies
leading_components=60
leading_components=60
Y = pd.DataFrame(ui_train)
I = Y.copy()
for col in I.columns:
    I[col] = I[col].apply(lambda x: 1 if x > 0 else 0)
U = np.random.normal(0, 0.01, [I.shape[0], leading_components])
V = np.random.normal(0, 0.01, [I.shape[1], leading_components])

def f(U, V):
    return np.sum(I.values*(np.dot(U, V.T)-Y.values)**2)
def dfu(U):
    return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)), V)
def dfv(V):
    return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)).T, U)


# gradient descent
N = 200
alpha = 0.001
pred = np.round(np.dot(U, V.T), decimals=2)
fobj = [f(U, V)]
rmsej = [rmse(pred, test)]
start = time.time()
for i in tqdm(range(N)):
    U = U - alpha*dfu(U)
    V = V - alpha*dfv(V)
    fobj.append(f(U, V))
    pred = np.round(np.dot(U, V.T), 2)
    rmsej.append(rmse(pred, test))
print('Time difference of {} mins'.format((time.time() - start) / 60))
fojb = np.array(fobj)
rmsej = np.array(rmsej)
path1 = pd.DataFrame({'itr': range(1, N+2), 'fobj': fobj, 'fobjp': fobj/max(fobj), 'rmse': rmsej, 'rmsep': rmsej/max(rmsej)})
path1gg = pd.melt(path1[["itr", "fobjp", "rmsep"]], id_vars=['itr'])
#print(path1.tail(1))


print(ggplot(path1gg, aes('itr', 'value', color = 'variable')) + geom_line())


# Create a rating based on time played
def game_hrs_density_p(pred, GAME=None, nclass=1, print_vals=True):
    game_dict = dict(games.values)
    t_GAME = GAME
    if not GAME:
        GAME = np.random.randint(0, games.shape[0])
    else:
        GAME = game_dict[GAME]
    game_data = pd.Series(pred[:, GAME])
    game_data = game_data[game_data > 0]

    # em algorithm
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
    not_purchased = (I - 1) % 2
    t_user = user
    user = user_dict[user]
    top_games = (pred_percentile*not_purchased).iloc[user]
    #     line = I_pred[user]
    #     for i, v in enumerate(line):
    #         if v == 1:
    #             top_games[i] = 0
    top_games = list(top_games.sort_values(ascending=False)[:20].index)
    if print_value:
        print('top {} recommended games for user {}: '.format(n, t_user))
        for i in range(n):
            print(i, ")", reverse_game_dict[top_games[i]])
    else:
        result = [t_user]
        for i in range(n):
            result.append(reverse_game_dict[top_games[i]])
        return result
top(20, 5250)

top_N = 20
result = []
for idx, user in tqdm(enumerate(users['user'].values)):
    result.append(top(top_N, user, False))
    if idx > 8212:
        break
df = pd.DataFrame(result)
columns = ['user'] + ['top_{}'.format(i+1) for i in range(top_N)]
df.columns = columns
df.to_csv('D:/Game-Recommendation-System/data/output_data/Collaborative_EM_output.csv', index=None)




