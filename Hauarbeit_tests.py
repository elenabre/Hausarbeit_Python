#%%test if exception is raised when no csv is given in Hausarbeit.py
import unittest
import Hausarbeit as H
import numpy as np


#%% test if find_nearest is working
np.shape(H.nearest_x_y_list)
#%%)
class TestFindNearest(unittest.TestCase):
    def test_find_nearest(self):
        array = np.array([1,2,3,4,5,6,7,8,9,10])
        '''define a test array'''
        self.assertEqual(H.find_nearest(array, 5), 5)
        '''test if the nearest value of 5 is 5'''
        self.assertEqual(H.find_nearest(array, 6), 6)
        '''test if the nearest value of 6 is 6'''
        self.assertEqual(H.find_nearest(array, 7), 7)
        '''test if the nearest value of 7 is 7'''
        self.assertEqual(H.find_nearest(array, 8), 8)
        '''test if the nearest value of 8 is 8'''
        self.assertEqual(H.find_nearest(array, 11), 10)
        '''test if the nearest value of 11 is 10'''
        self.assertEqual(H.find_nearest(array, 0), 1)
        '''test if the nearest value of 0 is 1'''
        self.assertEqual(H.find_nearest(array, -1), 1)
        '''test if the nearest value of -1 is 1'''

# test if Class Dev is working

#create test arrays
array1 = np.array(H.nearest_x_y_list)[:,:,1]
array1 = array1.ravel()
array1 = array1 + 1 
'''create array where deviation is 1'''
array2 = np.array(H.nearest_x_y_list)[:,:,2]
array2 = array2.ravel()
array2 = array2
'''create array where deviation is 0'''
array3 = np.array(H.nearest_x_y_list)[:,:,3]
array3 = array3.ravel()
array3 = array3 - 1
'''create array where deviation is -1, so should be 1'''
array4 = np.array(H.nearest_x_y_list)[:,:,4]
array4 = array4.ravel()
array4 = array4 + 2
class TestDev(unittest.TestCase):
    
# test if Class Dev is working
    def test_find_deviation(self):
        test1 = H.Dev()
        test1.find_deviation(H.nearest_x_y_list, 1, array1)
        '''use class Dev with array1'''
        test1 = np.array(test1.deviation)[1]
        self.assertEqual(test1, 1)
        '''as we added 1 to the array, the deviation should be 1'''
        test2 = H.Dev()
        test2.find_deviation(H.nearest_x_y_list, 2, array2)
        test2 = np.array(test2.deviation)[1]
        self.assertEqual(test2, 0)
        '''as we used the same array, the deviation should be 0'''
        test3 = H.Dev()
        test3.find_deviation(H.nearest_x_y_list, 3, array3)
        test3 = np.array(test3.deviation)[1]
        self.assertEqual(test3, 1)
        '''as we subtracted 1 from the array, the deviation should be 1 as we calculate absolute value'''
        test4 = H.Dev()
        test4.find_deviation(H.nearest_x_y_list, 4, array4)
        test4 = np.array(test4.deviation)[1]
        self.assertEqual(test4, 2)
        '''as we added 2 to the array, the deviation should be 2'''
    
 
# test Exception if no csv is given
class TestException(unittest.TestCase):
    def test_exception(self):
        with self.assertRaises(Exception):
            H.main()
            '''test if exception is raised when no csv is given in Hausarbeit.py'''

# test if exception is raised when no train.csv is given in Hausarbeit.py
class TestException2(unittest.TestCase):
    def test_exception2(self):
        with self.assertRaises(Exception):
            H.main('train.csv')
            '''test if exception is raised when no train.csv is given in Hausarbeit.py'''

# test if exception is raised when no ideal.csv is given in Hausarbeit.py
class TestException3(unittest.TestCase):
    def test_exception3(self):
        with self.assertRaises(Exception):
            H.main('ideal.csv')
            '''test if exception is raised when no ideal.csv is given in Hausarbeit.py'''

# test if exception is raised when no test.csv is given in Hausarbeit.py
class TestException4(unittest.TestCase):
    def test_exception4(self):
        with self.assertRaises(Exception):
            H.main('test.csv')
            '''test if exception is raised when no test.csv is given in Hausarbeit.py'''

        
# run all tests
if __name__ == '__main__':
    unittest.main()


