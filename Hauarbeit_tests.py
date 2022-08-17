#%%test if exception is raised when no csv is given in Hausarbeit.py
import unittest
import Hausarbeit as H
import numpy as np


# test if find_nearest is working
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

# test if exception is raised when no csv is given in Hausarbeit.py
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

        



if __name__ == '__main__':
    unittest.main()











        

# %%
