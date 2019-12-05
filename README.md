<style type="text/css"> .gist {width:500px; overflow:auto}  .gist .file-data {max-height: 500px;max-width: 500px;} </style>

# Recommendation System for Steam Game Store: An overview of recommender systems

## Team:<br/>
- Doo Hyodan, Department Of Information System, Hanyang University, ammissyouyou@gmail.com<br/>
- Audrey Germain, Computer Engineering, Department of Software Engineering and Computer Engineering, Polytechnique Montreal<br/>
- Geordan Jove, Aerospace Engineering, Department of Mechanical Engineering, Polytechnique Montreal<br/>

## Table of content

[I. Introduction](#introduction)<br/>
[II. Datasets](#dataset)<br/>
&nbsp;&nbsp;[a. User Dataset](#user)<br/>
&nbsp;&nbsp;[b. Game Dataset](#game)<br/>
[III. Methodology](#methodology)<br/>
&nbsp;&nbsp;[Collaborative Recommender](#collaborative)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;[a. Training and Test Datasets](#training-test)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;[b. Collaborative Recommender with ALS](#als)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;[c. Collaborative recommender with EM and SVD](#em)<br/>
&nbsp;&nbsp;[Content-based Recommender](#content-based)<br/>
[IV. Evaluation & Analysis](#evaluation-analysis)<br/>
[V. Related Work](#related-work)<br/>
[VI. Conclusion: Discussion](#conclusion)<br/>

## I. Introduction <a name="introduction"></a>
Like many young people, all member of this team have an interest in video games, more particularly in computer games.
Therefore, this project has for goal to build a recommender system for computer game.<br/>

This project was implemented as our final project for the course Introduction to Artificial Intelligence (ITE351) at Hanyang University.<br/>

For this project we are using the data from Steam the biggest video game digital distribution service for computer games.
We will be using some user data and game data, the datasets used will be explained in detail further in this blog.<br/>

The primary focus of this project is to build a recommendation system that will recommend games for each user based on
their preferences and gaming habits. In order to make our recommendation system the best possible, multiple algorithms

will be implemented and compared to make the recommendation the most relevant possible for each users.<br/>

There are three main types of recommendation system: collaborative filtering, content-based filtering and hybrid recommendation system. 
The collaborative filtering is based on the principle that if two people liked the same things in the past, 
if one of them like something the other is likely to like it too. 
The advantage of this filtering method is that the algorithm doesn’t need to understand or process the content 
of the items it recommends. The content-based filtering is based on the description of the items to recommend 
similar items and recommend items similar to what a user likes. 
Finally, the hybrid recommendation system consists of combining content based and collaborative filtering, 
either by using an algorithm that uses both or by combining the recommendation found by both methods. 
According to research combining both results is a better recommendation than using only one of them. <br/>

The data used for a recommendation system can be explicit, such as comment or rating, 
or implicit, such as behavior and events like order history, search logs, clicks, etc. 
The implicit data is harder to process because it’s hard to determine which information is useful and useless, 
but it’s easier to acquire than explicit data because the user doesn’t need to do anything more than 
using the website or app as usual. <br/>

In this project we implemented 2 collaboratives and 1 content based algorithm. 
We used implicite data from the users to implement them.

## II. Datasets <a name="dataset"></a>
For this project, 2 differents datasets are used. Both dataset are available for free on kaggle and are extracted from Steam.<br/>

### a. User Dataset <a name="user"></a>

[//]: # (User Dataset description)
The first dataset is the [user](https://www.kaggle.com/tamber/steam-video-games) dataset. It contains the user id, the game, the behavior and the amount of hours played.
So each row of the dataset represent the behavior (play or purchase) of a user towards a game.
The amount of hours played is also specify, the column contains 1 if it's a purchase.
The dataset contains a total of 200000 rows, including 5155 different games and 12393 different users.

[//]: # (Reformat with purchase/play columns)
The third dataset is a list we split out the purchase/play column into two columns. Because The raw 'purchase' row records 1 hour, Clearly this doesn't make any sense so we remove it during the process of splitting the column and calculate the correct 'play' hours.

| index |                                       game  |  user|       hrs|
| :---: | :------------------------------------------ | ---: | -------: |
|  1336 |                                       Dota 2|   484|  981684.6|
|  4257 |                              Team Fortress 2|  2323|  173673.3|
|  4788 |                                     Unturned|  1563|   16096.4|
|   981 |             Counter-Strike Global Offensive |  1412|  322771.6|
|  2074 |                       Half-Life 2 Lost Coast|   981|     184.4|
|   984 |                        Counter-Strike Source|   978|   96075.5|
|  2475 |                                Left 4 Dead 2|   951|   33596.7|
|   978 |                               Counter-Strike|   856|  134261.1|
|  4899 |                                     Warframe|   847|   27074.6|
|  2071 |                       Half-Life 2 Deathmatch|   823|    3712.9|
|  1894 |                                  Garry's Mod|   731|   49725.3|
|  4364 |                   The Elder Scrolls V Skyrim|   716|   70616.3|
|  3562 |                                    Robocraft|   689|    9096.6|
|   980 | Counter-Strike Condition Zero Deleted Scenes|   679|     418.2|
|   979 |                Counter-Strike Condition Zero|   679|    7950.0|
|  2142 |                            Heroes & Generals|   658|    3299.5|
|  2070 |                                  Half-Life 2|   639|    4260.3|
|  3825 |                   Sid Meier's Civilization V|   596|   99821.3|
|  4885 |                                  War Thunder|   590|   14381.6|
|  3222 |                                       Portal|   588|    2282.8|

As you can see Dota 2 has the highest number of players and the highest number of total hours played so undeniably the most popular game. Where as other games such as "Half-Life 2 Lost Coast" have 981 users but a total of 184.4 hours played. I expect this game is in most cases a free bundle game. 

[//]: # (Histogram 1: all users: play + purchase)
Then we count the number of users for each games, and output a histograph to make the data visualization. 
![Image text](https://github.com/AudreyGermain/Game-Recommendation-System/blob/master/plots/Histogram_AllUsersHrs.png)

[//]: # (Histogram 2: only users: play)
After we removed the users who just purchased the games but hasn't played. Some games fell from the top 20.
![Image text](https://github.com/AudreyGermain/Game-Recommendation-System/blob/master/plots/Histogram_UsersHrs.png)
Some Games like these add noise to the dataset. So that's one of the reasons we use EM algorithms to create rating system for the games.

[//]: # (Box plot)
In order to have a better understanding of the user data distribution and user's playing habits, a box plot is produced for the top 20 most played games.

![Image text](https://github.com/AudreyGermain/Game-Recommendation-System/blob/master/plots/boxplot_top_20_games.png?raw=true)

As we can see, the data distribution for each game considered is not symmetrical. Even more, 75% of data points for each game is in the range of the hundreds hours, with several games having very large outliers. We can see for example a user played more than 10,000 hours "Dota 2". Another interesting example, a user played almost 12,000 hours "Sid Meier's Civilization V".

### b. Game Dataset <a name="game"></a>
[//]: # (Game Dataset description)
The second dataset contains a list of [games](https://www.kaggle.com/trolukovich/steam-games-complete-dataset/version/1) and their descriptions. It contains the url (directed to Steam store),
the type of package (app, bundle…), the name of the game, a short description, recent reviews, all reviews, release date,
developper, publisher, popular tags (Gore, Action, Shooter, PvP…), game detail (Multi-player, Single-player, Full controller support…),
languages, achievements, genre (Action, Adventure, RPG, Strategy…), game description, description of mature content,
minimum requirement to run the game, recommended requirement, original price and price with discount.
There is a total of 51920 games in the dataset.<br/>

## III. Methodology <a name="methodology"></a>

We decided to use 3 differents algorithms to generate recommendation by user. We use 2 collaborative algorithm,
one using the ALS and one using the EM and SVD algorithms and we use one content-based algorithm.<br/>

### Collaborative Recommender <a name="collaborative"></a>

#### a. Training and Test Datasets <a name="training-test"></a>
[//]: # (Describe splitting of user dataset into training and testing)
To create a training and testing dataset, we started by combining the information about play and purchase in a single row,
in this new form, the columns are user ID, name of the game, amount of hours of play time, play (0 if never played
and 1 if played) and purchase (technically always 1), this created a total of 128804 rows. Then we extracted 20% of
all the rows (25761 rows) for the test dataset and kept the rest  (103043 rows) for the training dataset.<br/>

#### b. Collaborative Recommender with ALS <a name="als"></a>
This section describes a simple implementation of a collaborative filtering recommendation algorithm using matrix factorization with implicit data.
The work presented is based on the ["ALS Implicit Collaborative Filtering"](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe "ALS Implicit Collaborative Filtering") and the ["A Gentle Introduction to Recommender Systems with Implicit Feedback"](https://jessesw.com/Rec-System/ "A Gentle Introduction to Recommender Systems with Implicit Feedback") blog posts.

Collaborative filtering does not require any information about the items or the users in order to provide recommendation. It only uses the interactions between users and items expressed by some kind of rating.

The data used for this recommender system is that of the steam users described in [subsection II.a](#user). The data does not explicitly contains the rating or preference of users towards the games, but rather it is **implicitly** expressed by the amount of hours users played games.

The Alternating Least Squares (ALS) is the model used to fit the data and generate recommendations. The ALS model is already implemented in the [implicit](https://github.com/benfred/implicit) python library thanks to [Ben Frederickson](http://www.benfrederickson.com/fast-implicit-matrix-factorization/).  
As described on its documentation [here](https://implicit.readthedocs.io/en/latest/als.html), the ALS algorithm available through the implicit library is a Recommendation Model based on the algorithms described in the paper [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) with performance optimizations described in [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf). The advantage of using the implicit python library versus a manual implementation of the algorithm is the speed required to generate recommendation since the ALS model in the implicit library uses Cython allowing the parallelization of the code among threads.

**(Explain what's ALS and what it does)**

In order to generate recommendations, the class ImplicitCollaborativeRecommender is implemented in a python script. The code is available here below.

<script src="https://gist.github.com/g30rdan/d992457bf34607493c19341c96761387.js"></script>

(To complete)


#### c. Collaborative recommender with EM and SVD <a name="em"></a>

Becaused our recommendation system should take consideration the games hasn't been played. We could create a rating system for games based on distribution of playing hours. Such like hours of some free bad games could have a distribution under 2 hours. As following, We use the EM algorithm rather than percentiles to present the distribution. In the EM algorithm, We use 5 groups as 5 stars to distinguish the good from the bad.

```python
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
```

![Image text](https://github.com/AudreyGermain/Game-Recommendation-System/blob/master/plots/EM_SingleAnalysis.png)

According to the plot, we could see the there are most of the users of Fallout 4 distribute in 4-5 groups. However there are a few users quickly lost their interests. It make sense to request a refund for the game that have benn played less than 2 hours. As you can see EM algorithm does a great job finding the groups of people with similar gaming habits and would potentially rate the game in a similar way. 

It does have some trouble converging which iThis example will use a gradient descent approach to find optimal U and V matrices which retain the actual observations with predict the missing values by drawing on the information between similar users and games. I have chosen a learning rate of 0.001 and will run for 200 iterations tracking the RMSE. The objective function is the squared error between the actual observed values and the predicted values. The U and V matrices are initialised with a random draw from a ~N(0, 0.01) distibution. This may take a few minutes to run.sn't surprising however the resulting distributions look very reasonable. 

```python
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
```
   
A user-item matrix is created with the users being the rows and games being the columns. The missing values are set to zero. The observed values are the log hours for each observed user-game combination. The data was subset to games which have greater than 50 users and users which played the game for greater than 2 hours. This was chosen as 2 hours is the limit in which Steam will offer a return if you did not like the purchased game (a shout out to the Australian Competition and Consumer Commission for that one!).

```python
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
```
The basic SVD approach will perform matrix factorisation using the first 60 leading components. Since the missing values are set to 0 the factorisation will try and recreate them which is not quite what we want. For this example we will simply impute the missing observations with a mean value.

```python
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
```
![Image text](https://github.com/AudreyGermain/Game-Recommendation-System/blob/master/plots/SVD_Compare.png)
This example will use a gradient descent approach to find optimal U and V matrices which retain the actual observations with predict the missing values by drawing on the information between similar users and games. I have chosen a learning rate of 0.001 and will run for 200 iterations tracking the RMSE. The objective function is the squared error between the actual observed values and the predicted values. The U and V matrices are initialised with a random draw from a ~N(0, 0.01) distibution. This may take a few minutes to run.

![Image text](https://github.com/AudreyGermain/Game-Recommendation-System/blob/master/plots/EM_SVD_Analysis.png?raw=true)

 
### Content-based Recommender <a name="content-based"></a>

To generate the recommendation for each game, the following function is used. The input of the function is the title of
the game as a string and the cosine matrix (explained later) and the output is a list of recommended game title.<br/>

```python
def get_recommendations(title, cosine_sim):

	if title not in listGames:
    	    return []

	# Get the index of the game that matches the name
	idx = indices[title]

	# if there's 2 games or more with same name (game RUSH)
	if type(idx) is Series:
    	    return []

	# Get the pairwise similarity scores of all games with that game
	sim_scores = list(enumerate(cosine_sim[idx]))

	# Sort the games based on the similarity scores
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

	# Get the scores of the most similar games
	# (not the first one because this games as a score of 1 (perfect score) similarity with itself)
	sim_scores = sim_scores[1:n_recommendation + 1]

	# Get the games indices
	movie_indices = [i[0] for i in sim_scores]

	# Return the top most similar games
	return dataGames['name'].iloc[movie_indices].tolist()

```
The variable listGames is a list of all the games that are in both of the dataset (user and game dataset).
We use this because there is a lot of games in the game dataset that have never been played or purchased by any user,
so there's no use in considering them in the recommender and some of the games in the user dataset are not in the game dataset.
To maximize the amount of match between the game titles in the datasets we removed all symboles and spaces and put
every letters in lower case. We were able to find 3036 games in the game dataset that match some of the 5152 games that are in the user dataset.<br/>

The variable indices is a reverse map that use the name as key to get the index of each game in the cosine similarity matrix.
We make sure that the idx is not a Series, it can happen in the case where 2 different games have the same name (in our dataset 2 games have the name "RUSH").<br/>
```python
# Construct a reverse map of indices and game names
indices = Series(dataGames.index, index=dataGames['name']).drop_duplicates()
```
Then we get the similarity score of each games from the matrix and we order it from the most similar to the less similar.
Finally we just need to extract the amount of recommendation that we want and return the list.
The variable n_recommendation contains the amount of recommendation we want to get, we decided to set it to 20.<br/>

To generate the cosine similarity matrix we use the following code. First it calculate the matrix of frequency of each
words in the chosen column (column_name) of each of the games, then it calculate the cosine similarity function.<br/>

```python
# Compute the Cosine Similarity matrix using the column
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(dataGames[column_name])
cosine_sim_matrix = cosine_similarity(count_matrix, count_matrix)
```
To get the recommendation for each user, we implemented a function that combines the recommendations and get the
recommendation with the best reviews (extracted from the game dataset). This function takes the ID of each user,
the list of recommendation (the recommendation function explained previously is applied to all the games a user
has and a list of all the recommendations is made) and the list of all the game the user already has. The function
return a dataframe containing the user ID in the first column and then 20 column with the top recommendations.<br/>

```python
def make_recommendation_for_user(user_id, game_list, game_user_have):
	if type(game_list) is not list or len(game_list) == 0:
    	    # return empty one
    	    return DataFrame(data=[[user_id] + [""] * n_recommendation], columns=col_names)

	# get reviews of game recommendation, remove the games the user already has and order them by reviews
	recommendation_reviews = dataReviews.loc[dataReviews['name'].isin(game_list)]
	recommendation_reviews = recommendation_reviews.loc[~recommendation_reviews['name'].isin(game_user_have)]
	recommendation_reviews = recommendation_reviews.sort_values(by="percentage_positive_review", ascending=False)

	if len(recommendation_reviews.index) < n_recommendation:
    	    return DataFrame(data=[[user_id] + recommendation_reviews["name"].tolist() +
                           	   [""] * (n_recommendation - len(recommendation_reviews.index))],
                     	     columns=col_names)
	else:
    	    return DataFrame(data=[[user_id] + recommendation_reviews["name"].tolist()[0:n_recommendation]],
                     	     columns=col_names)
```
First, if the list of recommendation is empty (can happen if none of the game the user has are in the game dataset)
or not valid, a dataframe without recommendation is returned. If there is no problem with the recommendation list,
we get a dataframe of the name of the recommended games and the review (percentage of positive review) and we remove
the games that the user already has (no need to recommend a game the user already has). Then the recommendation are
ordered from the best review to the worst. If there is less recommendations then needed, empty spaces fill the rest of the column.
The reviews are used to order the recommendation since it's the easiest way to order them particularly considering that
not every games the user has produce recommendation because of the games in both datasets do not match totally like mentionned above.
If it wasn't because of that problem, we thought of taking into account the proportion of play time of each games to
recommend similar games to games that are most played. Using the reviews still ensure that the recommended games are
considered good in general by all the users.<br/>

To obtain the reviews, we had to do some manipulations on the review column in the game dataset to extract the 
percentage and other possibly useful information. We created the following script to do this and print the 
result in a CSV file. We read it from the content-based recommender script to get the reviews.

```python
for i, row in dataGames.iterrows():
    if type(row["all_reviews"]) == str:

        # extract % of positive reviews
        x = re.findall(r'- [0,1,2,3,4,5,6,7,8,9]*%', row["all_reviews"])
        if len(x) != 0:
            dataGames.at[i, 'percentage_positive_review'] = x[0].translate({ord(i): None for i in '- %'})

        # extract qualification of reviews
        reviewParse = row["all_reviews"].split(",")
        if 'user reviews' in reviewParse[0]:
            dataGames.at[i, 'review_qualification'] = ""
        else:
            dataGames.at[i, 'review_qualification'] = reviewParse[0]
```

We used the fact that all reviews follow this format:
"Mostly Positive,(11,481),- 74% of the 11,481 user reviews for this game are positive." 
to extract the information we wanted. 
We start by getting the percentage of good reviews by using regex to get the "- 74%" part of the reviews and we then keep the number only.
We also got the qualitative review by splitting the reviews at the comma and keeping the first one.
We ignore the qualification that contains the words 'user reviews' because it means not enough user reviewed the game
and the format is different. 

All the dataframe rows produced by the function 'make_recommendation_for_user' are combine and are printed in a CSV file.<br/>

The whole script is arranged in a function to let us run it with multiple different input.
The different input of the content based recommender are generated in a script that rearrange the data in an easy 
to use it for the algorithm, it correspond to the varible column_name in the code above.<br/>

To prepare the data for the content based recommender we started by selecting the information we thought would be the 
most useful to find similar games. We read the useful column using the following code.

```python
dataGames = read_csv(locationGamesFile, usecols=["name", "genre", "game_details", "popular_tags", "publisher", "developer"])
```

Like mentioned previously, we decided to only keep the games that were both in the game dataset and the user dataset.
We chose to do it this way because there is no point in recommending games the user don't have and it will affect the 
performance of the algorithm compare to the other too. Also the dataset was too big to create matrix of cosine similarity
since it took to much memory. To match the games from both dataset together, we created an ID for the games by removing
all symbols the weren't alphanumeric, removing all capital letters and removing all spaces using to following code. 
We did the same on the games in the user dataset.

```python
# remove spaces and special character from game name in both dataset
for i, row in dataGames.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["name"])
    clean = clean.lower()
    dataGames.at[i, 'ID'] = clean
```
After this, we found all the uniques ID from the user dataset and kept only the rows in the games dataset where the ID 
matched one of the ID in the user dataset. This way we were able to get 3036 games of the game dataset that matched 
some of the 5152 games from the user dataset. Without the ID we only were able to find 71 games that matched only 
using the names. Since we have less games in the new game dataset, the recommender system will not be able to find 
recommendation for every games in the user dataset. This will surly affect its performance.<br/>

With the new smaller game dataset, we made sure to remove the spaces from le useful column we chose to use.
By removing the spaces, we ensure that, for exemple, Steam Achievement and Steam Cloud dont get a match because they 
both contain Steam. Therefor we apply the following function to all the columns like this.

```python
def clean_data(x):
    if isinstance(x, str):
        return x.replace(" ", "")
    else:
        print(x)
        return x


usedGames.loc[:, 'genre'] = usedGames['genre'].apply(clean_data)
```

Finally, we created some custom columns by combining multiple column to try and find the combination of information 
that will give us the best recommendation system possible. 

```python
# create some column containing a mix of different information
usedGames["genre_publisher_developer"] = usedGames['genre'] + usedGames['publisher'] + usedGames['developer']
usedGames["genre_popular_tags_developer"] = usedGames['genre'] + usedGames['popular_tags'] + usedGames['developer']
usedGames["genre_popular_tags_game_details"] = usedGames['genre'] + usedGames['popular_tags'] + usedGames['game_details']
usedGames["genre_publisher_developer_game_details"] = usedGames['genre'] + usedGames['publisher'] + usedGames['developer'] + usedGames['game_details']
```

The results of the different columns will be compared in the Evaluation & Analysis section of this article.
## IV. Evaluation & Analysis <a name="evaluation-analysis"></a>

To compare the different algorithms we created a script that calculate the ratio of games the user has 
(in the test dataset) that are in the top 20 recommendations and the amount of games the user has in the
test dataset. The mean of the ratio for all users is the calculated. The ratio is a bit low because for some
user in the training dataset it was not possible to get the recommendations and some user don't have games
in the test dataset, in those cases the ratio will be 0. <br>

First of all, we compared the content based algorithm with different input. 
Those input are either column from the original dataset or a combination of different columns. 
In the following table we ca see the ratio for the different inputs. 
As we can see, the best algorithm is the one that uses the genre, publisher and developer of the game as input.
This version will be used to compare the other 2 algorithms.

Algorithm| Ratio
------------ | -------------
Popular tags| 0.6455%
Genre | 1.0847%
Genre, popular tags & developer | 0.6992%
Genre, popular tags & game details | 0.9144%
Genre, publisher & developer | 1.8198%
Genre, publisher, developer & game details | 1.6764%

We calculated the ratio in the same way for both of the collaborative algorithm, the result are in the following table.
As we can see, the collaborative recommender with ALS is the best one, followed by the content based recommender.
The performance of the collaborative recommendation system with EM and SVD is far behind the other 2.

Algorithm| Ratio
------------ | -------------
Collaborative with ALS| 2.6707%
Collaborative with EM and SVD | 0.3652%
Content-based (Genre, publisher & developer) | 1.8198%

## V. Related Work <a name="related-work"></a>

To understand what a recommender system is and the different types, we used [this article](https://marutitech.com/recommendation-engine-benefits/).<br/>

To manipulate the datasets, we used the [pandas](https://pandas.pydata.org/) library.<br/>

For the content-based recommender system we used the some code from the blog post
[Recommender Systems in Python: Beginner Tutorial](https://www.datacamp.com/community/tutorials/recommender-systems-python?fbclid=IwAR1fz9YLOgZ95KHwoLpgb-hTdV2MekujDGBngRTG3kYmBJYxwSK3UWvNJDg)
to implement the function that give the recommendation for each games.<br/>

For the EM algorithm we used the article
[Machine Learning with Python: Exception Maximization and Gaussian Mixture Models in Python](https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php).<br/>

## VI. Conclusion: Discussion <a name="conclusion"></a>

In conclusion, we implemented 3 different algorithm for recommendation system, one content-based and two collaborative,
one with the ALS algorithm and the other with the EM and SVD algorithm. The collaborative recommender with the ALS
algorithm seems to give the best recommendation based on our evaluation.<br/>

It would be interesting to create an hybrid recommender system using the collaborative recommender with the ALS 
algorithm and the content based algorithm using the genre, publisher and developer as input to see if we can make 
a recommender system even more effective then the 2 on their own.
