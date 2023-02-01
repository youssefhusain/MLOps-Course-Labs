import unittest
from src.app import sum  

class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum(2, 3), 5)
