import unittest
from app import sum, mult  

class TestMathFunctions(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum(2, 3), 5)
        self.assertEqual(sum(-1, 1), 0)
        self.assertEqual(sum(0, 0), 0)

    def test_mult(self):
        self.assertEqual(mult(2, 3), 6)
        self.assertEqual(mult(-1, 5), -5)
        self.assertEqual(mult(0, 10), 0)

if __name__ == '__main__':
    unittest.main()
