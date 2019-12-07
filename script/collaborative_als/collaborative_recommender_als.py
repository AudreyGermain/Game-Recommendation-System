import implicit
import pathlib
import pandas as pd
import scipy.sparse as sparse


class ImplicitCollaborativeRecommender:
    """
    Class developed to produce recommendations using implicit rating data and
    the recommendation model 'Alternating Least Squares' from the 'implicit'
    library.

    The 'Alternating Least Squares' available through the 'implicit' library
    is a Recommendation Model based on the algorithms described in the paper
    ‘Collaborative Filtering for Implicit Feedback Datasets’ with performance
    optimizations described in ‘Applications of the Conjugate Gradient Method
    for Implicit Feedback Collaborative Filtering’.
    Ref.: https://implicit.readthedocs.io/en/latest/als.html

    The implicit rating dataset to be used is should have the following
    structure:
      [user] [item] [implicit measure]
    """

    # Class attributes.
    # Internal column names
    __user_intl, __item_intl = 'user_internal', 'item_internal',
    __impl_intl = 'implicit_measure'

    # Original column names
    __user_o, __item_o, __impl_o = None, None, None

    # Data and lookup tables.
    data, lookup_users, lookup_items = None, None, None

    # Model and sparse matrices.
    __model, __m_user_item, __m_item_user = None, None, None

    def __init__(self, data_path=None):
        """
        Class initialization.
        If a path to the csv dataset is provided, the dataset is loaded and the
        ALS model is created. Otherwise, not.
        """

        if data_path is not None:
            self.load_data(data_path)

        if self.data is not None:
            self.load_model()

    def load_data(self, data_path):
        """
        Method to load data from input csv. Data are arranged in order to
        contain only the information that will be feed the ALS model:
           ['user_id'] ['item_id'] ['implicit measure']
        Original 'user' and 'item' information are replaced by codes.
        Two look up tables are generated in order to keep track of the
        'user'-'user_id' and 'item'-'item_id'. Information in these lookup
        tables is provided as strings.

        Note:
        The 'user_id' and 'item_id' used in the code refer to internal column
        names and not the 'user' or 'item' column names from the input dataset.
        """

        # Load training data.
        df_data = pd.read_csv(pathlib.Path(data_path))

        # Column numbers.
        col_user = df_data.columns[0]  # Name of column 'user'.
        col_item = df_data.columns[1]  # Name of column 'item'.
        col_impl = df_data.columns[2]  # Name of column 'implicit' measure.

        # Retrieve internal names for data columns.
        col_user_intl = self.__user_intl  # Internal name for 'user'.
        col_item_intl = self.__item_intl  # Internal name for 'item'.
        col_impl_intl = self.__impl_intl  # Internal name for 'implicit measure'.

        # Verify if NA data is present in dataset.
        print('\nNumber of NA data per column:')
        print(df_data.isna().sum(axis=0))

        # Convert 'user' and 'item' into numerical ID.
        df_data[col_user_intl] = df_data[col_user].astype('category').cat.codes
        df_data[col_item_intl] = df_data[col_item].astype('category').cat.codes

        # Create lookup tables for 'user_id - user' and 'item_id - item'.
        lookup_user = df_data[[col_user_intl, col_user]].drop_duplicates()
        lookup_user[col_user_intl] = lookup_user[col_user_intl].astype(str)
        lookup_user[col_user] = lookup_user[col_user].astype(int).astype(str)
        lookup_game = df_data[[col_item_intl, col_item]].drop_duplicates()
        lookup_game[col_item_intl] = lookup_game[col_item_intl].astype(str)

        # Clean dataframe with columns: 'user_id', 'item_id' and 'implicit
        # measure'.
        print(col_impl)
        print(col_impl_intl)
        df_data.rename(columns={col_impl: col_impl_intl}, inplace=True)
        df_data = df_data[[col_user_intl, col_item_intl, col_impl_intl]]

        # Verify all 'implicit measure' data considered is greater than zero.
        print('\nColumn \'{}\' statistics:'.format(col_impl))
        print(df_data[col_impl_intl].describe())

        # Assign results to class variables.
        self.data = df_data
        self.lookup_users = lookup_user
        self.lookup_items = lookup_game
        self.__user_o = col_user
        self.__item_o = col_item
        self.__impl_o = col_impl

    def load_model(self):
        """
        Method to create the ALS model using the 'implicit' library in order
        to produce recommendations based on the implicit rating data used. The
        dataset used corresponds to that loaded and arranged by the method
        'load_data'.

        Note:
        The implicit library expects data as a item-user matrix. Thus, two
        matrices containing the 'implicit measure' data are created:
             - Matrix for fitting the model (item-user)
             - Matrix to make recommendations (user-item)
            Ref.:
              https://medium.com/radon-dev/
              als-implicit-collaborative-filtering-5ed653ba39fe
        """

        # Retrieve internal names for data columns.
        col_user_intl = self.__user_intl
        col_item_intl = self.__item_intl
        col_impl_intl = self.__impl_intl

        if self.data is not None:

            df_data = self.data

            # Create sparse matrices: item-user and user-item.
            sparse_item_user = sparse.csr_matrix((df_data[col_impl_intl].astype(float),
                                                  (df_data[col_item_intl],
                                                   df_data[col_user_intl])))
            sparse_user_item = sparse.csr_matrix((df_data[col_impl_intl].astype(float),
                                                  (df_data[col_user_intl],
                                                   df_data[col_item_intl])))

            # Initialize the als model and fit it using the sparse item-user matrix
            model = implicit.als.AlternatingLeastSquares(factors=20,
                                                         regularization=0.1,
                                                         iterations=20)

            # Calculate the confidence by multiplying it the the defined alpha value.
            alpha_val = 15
            data_conf = (sparse_item_user * alpha_val).astype('double')

            # Fit data to the model
            model.fit(data_conf)

            # Assign results to class variables.
            self.__model = model
            self.__m_item_user = sparse_item_user
            self.__m_user_item = sparse_user_item

        else:
            self.__model = None
            self.__m_item_user = None
            self.__m_user_item = None

    def similar_items(self, items, n_similar):
        """
        Method to find the 'n' most similar items to the chosen 'item_id'.
        Results are returned as a dataframe:
          [user] [1] [2] [3] [4] .... [n]
        The 'items' input should be a list.
        """

        model = self.__model
        lookup_items = self.lookup_items
        n_similar = n_similar + 1

        # Retrieve column names.
        col_item_intl = self.__item_intl
        col_item_o = self.__item_o

        # Use implicit library methods to get similar items.
        output = []
        for item in items:
            item_id = lookup_items[col_item_intl]. \
                loc[lookup_items[col_item_o] == str(item)]

            if item_id.empty:
                item_names = [-999] * n_similar
            else:
                item_id = item_id.to_string(index=False).strip()
                similar = model.similar_items(int(item_id), n_similar)

                item_names = []
                for item_id_r, score in similar:
                    item_name = lookup_items[col_item_o]. \
                        loc[lookup_items[col_item_intl] == str(item_id_r)]. \
                        to_string(index=False).strip()
                    item_names.append(item_name)

            output.append(item_names)

        # Create dataframe to store similar items results.
        col_names = list(map(str, range(1, n_similar)))
        df_similar = pd.DataFrame(output, columns=[col_item_o] + col_names)

        return df_similar

    def recommend(self, users, n_recommendation):
        """
        Method to recommend 'n' items to the given 'users'.
        Results are returned as a dataframe:
          [user_id] [1] [2] [3] [4] .... [n]
        The 'users' input should be a list.
        """

        model = self.__model
        lookup_users = self.lookup_users
        lookup_items = self.lookup_items

        # Retrieve column names.
        col_user_intl = self.__user_intl
        col_user_o = self.__user_o
        col_item_intl = self.__item_intl
        col_item_o = self.__item_o

        # Use the implicit library recommend method.
        output = []
        for user in users:
            user_id = lookup_users[col_user_intl]. \
                loc[lookup_users[col_user_o] == str(user)]

            if user_id.empty:
                item_names = [-999] * n_recommendation
            else:
                user_id = user_id.to_string(index=False).strip()
                recommended = model.recommend(int(user_id),
                                              self.__m_user_item,
                                              n_recommendation,
                                              False)

                item_names = []
                for item_id, score in recommended:
                    item_name = lookup_items[col_item_o]. \
                        loc[lookup_items[col_item_intl] == str(item_id)]. \
                        to_string(index=False).strip()
                    item_names.append(item_name)

            output.append([user, *item_names])

        # Create dataframe to store recommendations.
        col_names = list(map(str, range(1, n_recommendation + 1)))
        df_recommendation = pd.DataFrame(output,
                                         columns=[col_user_o] + col_names)

        return df_recommendation


if __name__ == "__main__":
    # Get users from test data for which recommendations will be generated.
    test_location = r'../../data/model_data/steam_user_test.csv'
    df_test = pd.read_csv(test_location)
    list_users = df_test['user_id'].to_list()

    # Create collaborative recommender model (ALS).
    train_location = r'../../data/model_data/steam_user_train.csv'
    f_implicit = ImplicitCollaborativeRecommender(train_location)
    # df_sim = f_implicit.similar_items(['Dota 2', 'xxxxx', 'Fallout 4', 'Left 4 Dead 2'], 20)
    df_rec = f_implicit.recommend(list_users, 20)
    df_rec.to_csv(r'../../data/output_data/Collaborative_recommender_als_output.csv',
                  index=False)
