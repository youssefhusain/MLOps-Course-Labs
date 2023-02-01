import unittest
from app import sum, mult

class TestMathFunctions(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum(2, 3), 5)

    def test_mult(self):
        self.assertEqual(mult(2, 4), 8)

if __name__ == '__main__':
    unittest.main()
