# tests/test_max_profit.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.max_profit import max_profit_schedule

def test_examples():
    # Test cases from the problem description
    # profit check remains same
    assert max_profit_schedule(7)['profit'] == 3000
    assert max_profit_schedule(8)['profit'] == 4500
    assert max_profit_schedule(13)['profit'] == 16500

    # check multiple solutions for n=7
    res7 = max_profit_schedule(7)
    assert len(res7['solutions']) >= 1
    # Check that at least one solution corresponds to T:1 or P:1
    # Actually for n=7, T (5 units, earns 1500*(7-5)=3000) and P (4 units, earns 1000*(7-4)=3000) match.
    # So we expect 2 distinct solutions if our logic covers both.
    assert len(res7['solutions']) == 2

if __name__ == "__main__":
    test_examples()
    print("All tests passed!")
