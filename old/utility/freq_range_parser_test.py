import unittest
import freq_range_parser

class FreqRangeParserTest(unittest.TestCase):
    def test(self):
        parse = freq_range_parser.parse
        self.assertEqual(parse('100'), (100.0, 100.0, False))
        self.assertEqual(parse('-100'), (-100.0, -100.0, False))
        self.assertEqual(parse('100-200'), (100.0, 200.0, False))
        self.assertEqual(parse('10e1 - 20e1'), (100.0, 200.0, False))
        self.assertEqual(parse('-100-100'), (-100, 100, False))
        self.assertEqual(parse('-200--100'), (-200, -100, False))
        self.assertEqual(parse('-200--100'), (-200, -100, False))
        self.assertEqual(parse('100hz'), (100, 100, True))
        self.assertEqual(parse('100-200 Hz'), (100, 200, True))
        self.assertEqual(parse('10-20 khz'), (10000, 20000, True))
        self.assertEqual(parse('garbage'), None)

if __name__ == '__main__':
    unittest.main()

