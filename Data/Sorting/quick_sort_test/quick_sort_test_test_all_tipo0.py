import unittest
from zigzag_sort_test import TestZigZagSort
from rainbow_sort_test import TestRainbowSort
from harmony_sort_test import TestHarmonySort
from turbo_sort_test import TestTurboSort
from funky_sort_test import TestFunkySort
from galaxy_sort_test import TestGalaxySort
from magic_sort_test import TestMagicSort
from stellar_sort_test import TestStellarSort
from alpha_sort_test import TestAlphaSort
from funky_fresh_sort_test import TestFunkyFreshSort

def mega_suite():
    mega_suite = unittest.TestSuite()
    mega_suite.addTest(unittest.makeSuite(TestZigZagSort))
    mega_suite.addTest(unittest.makeSuite(TestRainbowSort))
    mega_suite.addTest(unittest.makeSuite(TestHarmonySort))
    mega_suite.addTest(unittest.makeSuite(TestTurboSort))
    mega_suite.addTest(unittest.makeSuite(TestFunkySort))
    mega_suite.addTest(unittest.makeSuite(TestGalaxySort))
    mega_suite.addTest(unittest.makeSuite(TestMagicSort))
    mega_suite.addTest(unittest.makeSuite(TestStellarSort))
    mega_suite.addTest(unittest.makeSuite(TestAlphaSort))
    mega_suite.addTest(unittest.makeSuite(TestFunkyFreshSort))
    return mega_suite

if __name__ == "__main__":
    executor = unittest.TextTestRunner(verbosity=2)
    executor.run(mega_suite()) 