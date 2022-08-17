#%% Import libraries
from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
import math

#%% We have train and ideal data. To find the four best fits with least square 
# from ideal and train we first need to read the data and store it in a database.
# create a class to read excel and store data in database
class DB:
    '''create a class to read excel and store data in database'''
    def read_data(self, table_name):
        '''read data from excel and store it in database'''
        self.table_name = table_name
        self.data = pd.read_csv(self.table_name + '.csv')
        # create a database engine
        self.engine = create_engine('sqlite:///' + self.table_name + '.db')
        # create all tables in the database
        Base = declarative_base()
        x = Column('x', Integer, primary_key=True)
        '''define x as primary key'''
        Base.metadata.create_all(self.engine)
        # create a session to connect to the database
        self.data.to_sql(con=self.engine, name=self.table_name, if_exists='replace', index=False)
        '''define name and replace existing database'''
  
# create object train and ideal to read excel and store data in database
train = DB()
train.read_data('train')
ideal = DB()
ideal.read_data('ideal')

# Exception if no train odr ideal csv is given in Hausarbeit.py
class NoFileFound(Exception):
    pass
try:
    train.table_name = open('train.csv')
except FileNotFoundError:
    raise NoFileFound('train.csv not found')
    '''raise exception if train.csv is not found'''
try:
    ideal.table_name = open('ideal.csv')
except FileNotFoundError:
    raise NoFileFound('ideal.csv not found')
    '''raise exception if ideal.csv is not found'''

# Exception if train has not five columns
def train_exception():
    '''exception if train has not five columns'''
    with self.assertRaises(Exception('train.csv has not five columns')):
        train.data = train.data.drop('x', axis=1)

#%% create list of train y-columns
train_list = train.data.columns.tolist()
train_list.pop(0)
'''skip column x to only see results for y-columns'''

#%% create an empty dictionary to store the four ideal functions with the least square sum of the y-deviation with the train data.
dict_four_ideal = {'train_datasets': train_list, 'ideal_results': [], 'sqrt_y_dev': [], 'max_dev': [], 'max_dev_sqrt': []}

# create a for loop to find the four ideal functions with the least square sum of the y-deviation
for y in train_list:
    # create array of train data
    train_array = np.array(train.data[y])
    # create a list of the range from the 50 ideal functions, starting with column 1
    ideal_range = list(range(1, 51, 1))

    # create empty lists to store the data for the ideal functions
    ideal_results_list = []
    y_dev_array_sqrt_sum_list = []
    max_dev_list = []
    max_dev_sqrt_list = []
    # create a for loop to find the ideal functions and store the data in the lists
    for i in ideal_range:
        # define the ideal function for y{i} and store result in results-list
        ideal_results = f'y{i}'
        ideal_results_list.append(ideal_results)
        # create array of ideal data
        ideal_array = np.array(ideal.data[ideal_results])

        # calculate the y-deviation between the ideal function and the train data
        y_dev_array = ideal_array - train_array
        # calculate the sum of the square root of the y-deviation
        y_dev_array = np.absolute(y_dev_array)
        y_dev_array_sqrt = np.sqrt(y_dev_array)
        y_dev_array_sqrt_sum = np.sum(y_dev_array_sqrt)
        # store the square root of the sum of the y-deviation in the list
        y_dev_array_sqrt_sum_list.append(y_dev_array_sqrt_sum)
        # calculate the maximum of the y-deviation
        max_dev = np.amax(y_dev_array)
        # store the maximum of the y-deviation in the list
        max_dev_list.append(max_dev)
        # calculate the square root of the maximum of the y-deviation
        max_dev_sqrt = np.sqrt(max_dev)
        # store the square root of the maximum of the y-deviation in the list
        max_dev_sqrt_list.append(max_dev_sqrt)
    # find the least square of the ideal functions and store the data in the dictionary
    min_sqrt_dev = min(y_dev_array_sqrt_sum_list)
    dict_four_ideal['sqrt_y_dev'].append(min_sqrt_dev)
    # create an index to search for the ideal functions
    min_sqrt_dev_index = y_dev_array_sqrt_sum_list.index(min_sqrt_dev)
    # store the ideal functions in the dictionary
    dict_four_ideal['ideal_results'].append(ideal_results_list[min_sqrt_dev_index])
    # store the maximum of the y-deviation in the dictionary
    dict_four_ideal['max_dev'].append(max_dev_list[min_sqrt_dev_index])
    # store the square root of the maximum of the y-deviation in the dictionary
    dict_four_ideal['max_dev_sqrt'].append(max_dev_sqrt_list[min_sqrt_dev_index])

# create a dataframe from the dictionary
df_four_ideal = pd.DataFrame(dict_four_ideal)
print(df_four_ideal)

#%% get the values of ideal.data where df_four_ideal.ideal_results is the same as the column in ideal.data and store the values in a new dataframe
df_four_ideal_data = ideal.data[df_four_ideal.ideal_results]
# plot the ideal_results
x_ideal = ideal.data.x
ideal_1 = df_four_ideal_data.iloc[:,0]
'''y column of first ideal function'''
train_y1 = train.data.y1
ideal_2 = df_four_ideal_data.iloc[:,1]
'''y column of second ideal function'''
train_y2 = train.data.y2
ideal_3 = df_four_ideal_data.iloc[:,2]
'''y column of third ideal function'''
train_y3 = train.data.y3
ideal_4 = df_four_ideal_data.iloc[:,3]
'''y column of fourth ideal function'''
train_y4 = train.data.y4

# create four plots for the ideal functions
class Four_Plot():
    '''create four plots with two functions each'''
    def __init__(self, x, x1, graph1_1, graph1_2, x2, graph2_1, graph2_2, x3, graph3_1, graph3_2, x4, graph4_1, graph4_2, legend_elements):
        '''define two functions for each plot, labels for each function and legend elements'''
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        axs[0, 0].plot(x1, graph1_1, 'b')
        axs[0,0].plot(x, graph1_2, 'r')
        axs[0,0].set_title(df_four_ideal.ideal_results[0])
        axs[0,0].legend(handles=legend_elements, loc='lower right')
        axs[0,1].plot(x2, graph2_1, 'b')
        axs[0,1].plot(x, graph2_2, 'r')
        axs[0,1].set_title(df_four_ideal.ideal_results[1])
        axs[0,1].legend(handles=legend_elements, loc='lower right')
        axs[1,0].plot(x3, graph3_1, 'b')
        axs[1,0].plot(x, graph3_2, 'r')
        axs[1,0].set_title(df_four_ideal.ideal_results[2])
        axs[1,0].legend(handles=legend_elements, loc='lower right')
        axs[1,1].plot(x4, graph4_1, 'b')
        axs[1,1].plot(x, graph4_2, 'r')
        axs[1,1].set_title(df_four_ideal.ideal_results[3])
        axs[1,1].legend(handles=legend_elements, loc='lower right')
        plt.show()
        

    def __call__(self, x, x1, graph1_1, graph1_2, x2, graph2_1, graph2_2, x3, graph3_1, graph3_2, x4, graph4_1, graph4_2, legend_elements):
        self.__init__(x, x1, graph1_1, graph1_2, x2, graph2_1, graph2_2, x3, graph3_1, graph3_2, x4, graph4_1, graph4_2, legend_elements)

# create a new instance of the class
# plot the ideal functions
legend_elements = [Line2D([0], [0], color='b', label='Train Function'),
                   Line2D([0], [0], color='r', label='Ideal Function')]
four_plot1 = Four_Plot(x_ideal, x_ideal, train_y1, ideal_1, x_ideal, train_y2, ideal_2, x_ideal, train_y3, ideal_3, x_ideal, train_y4, ideal_4, legend_elements)


#%% analyze with test.csv. dict_four_ideal has four ideal functions. Every point from test.csv is compared to the ideal functions. 
# The ideal function with the least deviation is the ideal function for the point. 
# Every test point that has a higher deviation than the deviation of the ideal function with the train function by the factor of the square root of 2 gets dropped.
data_test = pd.read_csv('test.csv')
# Exception handling if test.csv is not found
try:
    data_test
except NameError:
    print('test.csv not found')

# Exception handling if test.csv has not two columns
try:
    data_test.shape[1] == 2
except NameError:
    print('test.csv has not two columns')


#Create dataframe with all ideal functions to compare it to test.csv
all_ideal_functions = pd.DataFrame().assign(x=x_ideal, y1=ideal_1, y2=ideal_2, y3=ideal_3, y4=ideal_4)
# create arrays for test data
test_x_array = np.array(data_test.x)
test_y_list = data_test.y.tolist()
test_y_array = np.array(test_y_list)
# create function to find the same or nearest values from test and ideal data
def find_nearest(array, value):
    '''compare array to a value'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    '''we look for the lowest difference between the array and the value - needs to be absolute value'''
    return array[idx]

# find same or nearest x and y value for test and ideal values through for loop and store in a list
nearest_x_list = []
nearest_x_y_list = []
for x in test_x_array:
    nearest_x = find_nearest(all_ideal_functions.x, x)
    '''use predefined function to find the nearest x value'''
    nearest_x_list.append(nearest_x)
    # create new dataframe with the nearest x values and ideal functions, create a list to find the nearest y value
    nearest_x_y = pd.DataFrame().assign(x=all_ideal_functions.x[all_ideal_functions.x == nearest_x], y1=all_ideal_functions.y1[all_ideal_functions.x == nearest_x], y2=all_ideal_functions.y2[all_ideal_functions.x == nearest_x], y3=all_ideal_functions.y3[all_ideal_functions.x == nearest_x], y4=all_ideal_functions.y4[all_ideal_functions.x == nearest_x])
    nearest_x_y_list.append(nearest_x_y)

# create arrays for each ideal function where the nearest x value was found 
nearest_x_y_array = np.array(nearest_x_y_list)
nearest_x_y1_array = np.array(nearest_x_y_list)[:,:,1]
'''second column of nearest_x_y_list is the first ideal y value'''
nearest_x_y1_array = nearest_x_y1_array.ravel()
'''ravel() flattens the array'''
nearest_x_y2_array = np.array(nearest_x_y_list)[:,:,2]
nearest_x_y2_array = nearest_x_y2_array.ravel()
nearest_x_y3_array = np.array(nearest_x_y_list)[:,:,3]
nearest_x_y3_array = nearest_x_y3_array.ravel()
nearest_x_y4_array = np.array(nearest_x_y_list)[:,:,4]
nearest_x_y4_array = nearest_x_y4_array.ravel()
# calculate the deviation of each ideal y and test y value and store as new array
dev_y_1_array = nearest_x_y1_array - test_y_array
dev_y_2_array = nearest_x_y2_array - test_y_array
dev_y_3_array = nearest_x_y3_array - test_y_array
dev_y_4_array = nearest_x_y4_array - test_y_array
# get the absolute value of the deviation
dev_y_1_array = np.absolute(dev_y_1_array)
dev_y_2_array = np.absolute(dev_y_2_array)
dev_y_3_array = np.absolute(dev_y_3_array)
dev_y_4_array = np.absolute(dev_y_4_array)
#create dataframe with all deviation data
dev_df = pd.DataFrame().assign(dev_y_1=dev_y_1_array, dev_y_2=dev_y_2_array, dev_y_3=dev_y_3_array, dev_y_4=dev_y_4_array)
# find min value column and min value column name and add it to dataframe
c = dev_df.columns
dev_df[c] = dev_df[c].apply(lambda x: pd.to_numeric(x))
'''convert to numeric'''
dev_df = dev_df.assign(min_dev = dev_df[c].min(axis=1), min_dev_name = dev_df[c].idxmin(axis=1))
'''find min deviation of all deviation in each row and the column name of this min deviation'''
# add x and y test values to dataframe
dev_df = dev_df.assign(x=test_x_array, y=test_y_array)
print(dev_df)

#%% compare deviation of four ideal with deviation of test data and store the values that do not
# exceed the four ideal daviation by the factor of sqrt(2) in a new dataframe
test_result = pd.DataFrame()
for i in dev_df.index:
    if dev_df.min_dev_name[i] == 'dev_y_1':
        '''if the min deviation is the first ideal deviation, calulate if the deviation is lower than the ideal deviation by the factor of sqrt(2)'''
        if dev_df.dev_y_1[i] <= df_four_ideal.max_dev[0]*math.sqrt(2):
            '''first column of max_dev is the max deviation of the first ideal function'''
            test_result = test_result.append(dev_df.loc[i])
            '''append the row with the lowest deviation to the new dataframe'''
    elif dev_df.min_dev_name[i] == 'dev_y_2':
        if dev_df.dev_y_2[i] <= df_four_ideal.max_dev[1]*math.sqrt(2):
            test_result = test_result.append(dev_df.loc[i])
    elif dev_df.min_dev_name[i] == 'dev_y_3':
        if dev_df.dev_y_3[i] <= df_four_ideal.max_dev[2]*math.sqrt(2):
            test_result = test_result.append(dev_df.loc[i])
    elif dev_df.min_dev_name[i] == 'dev_y_4':
        if dev_df.dev_y_4[i] <= df_four_ideal.max_dev[3]*math.sqrt(2):
            test_result = test_result.append(dev_df.loc[i])

# Exception handling for if there is no data where the deviation is lower than the ideal deviation by the factor of sqrt(2)
try:
    test_result
except NameError:
    print('No data where the deviation is lower than the ideal deviation by the factor of sqrt(2)')

#%% create dataframe with x, y, min_dev and min_dev_name values of test_result, where for each min_dev_name value the y of ideal is shown
test_result_table = pd.DataFrame().assign(x=test_result.x, y=test_result.y, min_dev=test_result.min_dev, min_dev_name=test_result.min_dev_name,)
'''use the min_dev and min_dev_name values of test_result to find the y value of the ideal function that has the lowest deviation and drop the other columns'''
Results = {"dev_y_1": df_four_ideal.ideal_results[0], "dev_y_2": df_four_ideal.ideal_results[1], "dev_y_3": df_four_ideal.ideal_results[2], "dev_y_4": df_four_ideal.ideal_results[3]}
'''use Results to get the cideal y function column instead of 1, 2, 3, 4'''
test_result_table = test_result_table.assign(ideal_y=test_result_table.min_dev_name.map(Results))
'''map the min_dev_name values to the ideal_y values'''
test_result_table.drop(columns=['min_dev_name'], inplace=True)
'''drop the min_dev_name column'''      
print(test_result_table)

# Exception handling if no ideal y value is found
try:
    test_result_table
except NameError:
    print('No ideal y value is found')


#%% store the test_result_table in database through sqlalchemy
engine = create_engine('sqlite:///test.db')    
test_result_table.to_sql('test', engine, if_exists='replace', index=False)

#%% plot x and y values of test_result_table with each associated ideal_y function
class test_points_to_list_y(object):
    def __init__(self, y):
        self.list = []
        for i in test_result_table.index:
            if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[y]:
                self.list.append(test_result_table.y[i])
        

test_y1 = test_points_to_list_y(0)
test_y1 = test_y1.list
test_y2 = test_points_to_list_y(1)
test_y2 = test_y2.list
test_y3 = test_points_to_list_y(2)
test_y3 = test_y3.list
test_y4 = test_points_to_list_y(3)
test_y4 = test_y4.list

class test_points_to_list_x(object):
    def __init__(self, x):
        self.list = []
        for i in test_result_table.index:
            if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[x]:
                self.list.append(test_result_table.x[i])
                
test_x1 = test_points_to_list_x(0)
test_x1 = test_x1.list
test_x2 = test_points_to_list_x(1)
test_x2 = test_x2.list
test_x3 = test_points_to_list_x(2)
test_x3 = test_x3.list
test_x4 = test_points_to_list_x(3)
test_x4 = test_x4.list

#%% plot x and y values of test_result_table with each associated ideal_y function
legend_elements = [Line2D([0], [0], color='r', lw=2, label='Ideal Function'),
                   Line2D([0], [0], marker='o', color='w', label='Test Point',
                          markerfacecolor='b', markersize=10)]
'''create a legend element for the ideal function and test point'''
four_plot2 = Four_Plot(x_ideal, test_x1, test_y1, ideal_1, test_x2, test_y2, ideal_2, test_x3, test_y3, ideal_3, test_x4, test_y4, ideal_4, legend_elements)
'''create a Four_Plot object with the ideal functions and test points'''
four_plot2.plot_four()
'''plot the ideal functions and test points'''
four_plot2.show_plot()

'''create a Four_Plot object with the test_points_to_list object and the ideal_function objects'''






#%%
fig, axs = plt.subplots(2, 2, figsize=(10,10))
'''create a figure with 2x2 subplots'''
legend_elements = [Line2D([0], [0], color='r', lw=2, label='Ideal Function'),
                   Line2D([0], [0], marker='o', color='w', label='Test Point',
                          markerfacecolor='b', markersize=10)]
'''create a legend element for the ideal function and test point'''
axs[0, 0].plot(x_ideal, ideal_1, 'r')
'''plot the ideal function on the first subplot'''
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[0]:
        axs[0, 0].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
        '''Plot all test points that are associated with the first ideal function on the first subplot'''
axs[0, 0].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[0]))
'''set the title of the first subplot to the ideal_y column name of the first ideal function'''
axs[0, 0].legend(handles=legend_elements, loc='lower right')
'''add the legend to the first subplot'''
axs[0, 1].plot(x_ideal, ideal_2, 'r')
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[1]:
        axs[0, 1].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
axs[0, 1].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[1]))
axs[0, 1].legend(handles=legend_elements, loc='lower right')
axs[1, 0].plot(x_ideal, ideal_3, 'r')
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[2]:
        axs[1, 0].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
axs[1, 0].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[2]))
axs[1, 0].legend(handles=legend_elements, loc='lower right')
axs[1, 1].plot(x_ideal, ideal_4, 'r')
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[3]:
        axs[1, 1].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
axs[1, 1].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[3]))
axs[1, 1].legend(handles=legend_elements, loc='lower right')

plt.show()



# %%
