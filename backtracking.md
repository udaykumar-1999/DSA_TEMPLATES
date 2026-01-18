# Backtracking, Permutations & Combinations: Complete Guide

## What is Backtracking?

Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally, removing solutions that fail to satisfy constraints (backtrack), and continuing with other candidates.

Think of it like exploring a maze - you try a path, and if it leads to a dead end, you backtrack to the last decision point and try a different path.

## Core Principles

**Backtracking Process:**
1. **Choose** - Make a choice from available options
2. **Explore** - Recursively explore that choice
3. **Unchoose (Backtrack)** - Undo the choice and try next option

**When to Use:**
- Generate all possible solutions
- Find all combinations/permutations
- Solve constraint satisfaction problems
- Explore all paths in decision tree

## Backtracking Template

### Basic Template

```python
def backtrack(path, choices):
    # Base case: solution found
    if is_valid_solution(path):
        result.append(path.copy())
        return
    
    # Try each choice
    for choice in choices:
        # 1. Make choice
        path.append(choice)
        
        # 2. Explore (recurse)
        backtrack(path, get_next_choices(choice))
        
        # 3. Undo choice (backtrack)
        path.pop()
```

### Template with Pruning

```python
def backtrack(path, choices, start):
    # Base case
    if meets_condition(path):
        result.append(path.copy())
        return
    
    for i in range(start, len(choices)):
        # Pruning: skip invalid choices
        if not is_valid(choices[i], path):
            continue
        
        # Make choice
        path.append(choices[i])
        
        # Explore
        backtrack(path, choices, i + 1)  # or i for repeats
        
        # Undo choice
        path.pop()
```

## Pattern 1: Permutations

### Permutations - All arrangements of elements where order matters.

**Formula:** n! (n factorial)
**Example:** [1, 2, 3] → [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]

### Example 1: Basic Permutations

**Problem:** Generate all permutations of distinct numbers.

```python
def permute(nums):
    """
    Time: O(n! × n), Space: O(n) for recursion stack
    """
    result = []
    
    def backtrack(path, remaining):
        # Base case: used all numbers
        if not remaining:
            result.append(path.copy())
            return
        
        for i in range(len(remaining)):
            # Choose
            path.append(remaining[i])
            
            # Explore: recurse with remaining elements
            backtrack(path, remaining[:i] + remaining[i+1:])
            
            # Unchoose
            path.pop()
    
    backtrack([], nums)
    return result

# Example usage
print(permute([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### Example 2: Permutations (Using Visited Array)

**Problem:** More efficient approach using visited tracking.

```python
def permute_visited(nums):
    """
    Time: O(n! × n), Space: O(n)
    More efficient - no list slicing
    """
    result = []
    n = len(nums)
    visited = [False] * n
    
    def backtrack(path):
        if len(path) == n:
            result.append(path.copy())
            return
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Choose
            path.append(nums[i])
            visited[i] = True
            
            # Explore
            backtrack(path)
            
            # Unchoose
            path.pop()
            visited[i] = False
    
    backtrack([])
    return result

# Example usage
print(permute_visited([1, 2, 3]))
```

### Example 3: Permutations with Duplicates

**Problem:** Generate permutations when array has duplicates.

```python
def permuteUnique(nums):
    """
    Time: O(n! × n), Space: O(n)
    Handle duplicates by sorting and skipping
    """
    result = []
    nums.sort()  # Sort to group duplicates
    visited = [False] * len(nums)
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path.copy())
            return
        
        for i in range(len(nums)):
            if visited[i]:
                continue
            
            # Skip duplicate: if current equals previous AND previous not used
            if i > 0 and nums[i] == nums[i-1] and not visited[i-1]:
                continue
            
            path.append(nums[i])
            visited[i] = True
            
            backtrack(path)
            
            path.pop()
            visited[i] = False
    
    backtrack([])
    return result

# Example usage
print(permuteUnique([1, 1, 2]))
# [[1,1,2], [1,2,1], [2,1,1]]
```

### Example 4: Next Permutation

**Problem:** Find next lexicographically greater permutation.

```python
def nextPermutation(nums):
    """
    Time: O(n), Space: O(1)
    In-place modification
    """
    n = len(nums)
    
    # 1. Find first decreasing element from right
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        # 2. Find element just larger than nums[i]
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        
        # 3. Swap
        nums[i], nums[j] = nums[j], nums[i]
    
    # 4. Reverse from i+1 to end
    nums[i + 1:] = reversed(nums[i + 1:])

# Example usage
nums = [1, 2, 3]
nextPermutation(nums)
print(nums)  # [1, 3, 2]
```

### Example 5: Permutation Sequence (Kth Permutation)

**Problem:** Find kth permutation of [1, 2, ..., n].

```python
def getPermutation(n, k):
    """
    Time: O(n²), Space: O(n)
    Direct construction without generating all
    """
    import math
    
    nums = list(range(1, n + 1))
    k -= 1  # Convert to 0-indexed
    result = []
    
    for i in range(n, 0, -1):
        factorial = math.factorial(i - 1)
        index = k // factorial
        result.append(str(nums[index]))
        nums.pop(index)
        k %= factorial
    
    return ''.join(result)

# Example usage
print(getPermutation(3, 3))  # "213"
print(getPermutation(4, 9))  # "2314"
```

## Pattern 2: Combinations

### Combinations - Selections where order doesn't matter.

**Formula:** C(n, k) = n! / (k! × (n-k)!)
**Example:** Choose 2 from [1,2,3] → [1,2], [1,3], [2,3]

### Example 1: Basic Combinations

**Problem:** Generate all combinations of k numbers from 1 to n.

```python
def combine(n, k):
    """
    Time: O(C(n,k) × k), Space: O(k)
    """
    result = []
    
    def backtrack(start, path):
        # Base case: combination complete
        if len(path) == k:
            result.append(path.copy())
            return
        
        # Pruning: if not enough elements left
        needed = k - len(path)
        remaining = n - start + 1
        if remaining < needed:
            return
        
        for i in range(start, n + 1):
            # Choose
            path.append(i)
            
            # Explore: start from i+1 to avoid duplicates
            backtrack(i + 1, path)
            
            # Unchoose
            path.pop()
    
    backtrack(1, [])
    return result

# Example usage
print(combine(4, 2))
# [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
```

### Example 2: Combination Sum (Unlimited Use)

**Problem:** Find all combinations that sum to target (can reuse elements).

```python
def combinationSum(candidates, target):
    """
    Time: O(n^(target/min)), Space: O(target/min)
    Elements can be reused
    """
    result = []
    candidates.sort()  # Optional: helps with pruning
    
    def backtrack(start, path, total):
        if total == target:
            result.append(path.copy())
            return
        
        if total > target:
            return
        
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            
            # Can reuse same element: start from i
            backtrack(i, path, total + candidates[i])
            
            path.pop()
    
    backtrack(0, [], 0)
    return result

# Example usage
print(combinationSum([2, 3, 6, 7], 7))
# [[2,2,3], [7]]
```

### Example 3: Combination Sum II (No Reuse, With Duplicates)

**Problem:** Find combinations that sum to target (no reuse, array has duplicates).

```python
def combinationSum2(candidates, target):
    """
    Time: O(2^n), Space: O(n)
    Each element used at most once
    """
    result = []
    candidates.sort()
    
    def backtrack(start, path, total):
        if total == target:
            result.append(path.copy())
            return
        
        if total > target:
            return
        
        for i in range(start, len(candidates)):
            # Skip duplicates at same level
            if i > start and candidates[i] == candidates[i-1]:
                continue
            
            path.append(candidates[i])
            
            # Move to next index: i+1 (no reuse)
            backtrack(i + 1, path, total + candidates[i])
            
            path.pop()
    
    backtrack(0, [], 0)
    return result

# Example usage
print(combinationSum2([10, 1, 2, 7, 6, 1, 5], 8))
# [[1,1,6], [1,2,5], [1,7], [2,6]]
```

### Example 4: Combination Sum III

**Problem:** Find k numbers that add up to n (only use 1-9, each at most once).

```python
def combinationSum3(k, n):
    """
    Time: O(C(9,k)), Space: O(k)
    """
    result = []
    
    def backtrack(start, path, total):
        if len(path) == k:
            if total == n:
                result.append(path.copy())
            return
        
        # Pruning
        if total > n:
            return
        
        for i in range(start, 10):
            path.append(i)
            backtrack(i + 1, path, total + i)
            path.pop()
    
    backtrack(1, [], 0)
    return result

# Example usage
print(combinationSum3(3, 7))
# [[1,2,4]]
print(combinationSum3(3, 9))
# [[1,2,6], [1,3,5], [2,3,4]]
```

## Pattern 3: Subsets (Power Set)

### Subsets - All possible combinations of any length.

**Formula:** 2^n subsets
**Example:** [1,2] → [], [1], [2], [1,2]

### Example 1: Basic Subsets

**Problem:** Generate all subsets of distinct elements.

```python
def subsets(nums):
    """
    Time: O(2^n × n), Space: O(n)
    """
    result = []
    
    def backtrack(start, path):
        # Add current subset to result
        result.append(path.copy())
        
        for i in range(start, len(nums)):
            # Choose
            path.append(nums[i])
            
            # Explore
            backtrack(i + 1, path)
            
            # Unchoose
            path.pop()
    
    backtrack(0, [])
    return result

# Example usage
print(subsets([1, 2, 3]))
# [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### Example 2: Subsets with Duplicates

**Problem:** Generate subsets when array has duplicates.

```python
def subsetsWithDup(nums):
    """
    Time: O(2^n × n), Space: O(n)
    """
    result = []
    nums.sort()  # Group duplicates
    
    def backtrack(start, path):
        result.append(path.copy())
        
        for i in range(start, len(nums)):
            # Skip duplicates at same level
            if i > start and nums[i] == nums[i-1]:
                continue
            
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# Example usage
print(subsetsWithDup([1, 2, 2]))
# [[], [1], [1,2], [1,2,2], [2], [2,2]]
```

### Example 3: Subsets (Iterative Approach)

**Problem:** Generate subsets iteratively using bit manipulation concept.

```python
def subsets_iterative(nums):
    """
    Time: O(2^n × n), Space: O(1) excluding output
    """
    result = [[]]
    
    for num in nums:
        # Add current number to all existing subsets
        result += [subset + [num] for subset in result]
    
    return result

# Alternative: Bit Manipulation
def subsets_bitmask(nums):
    """
    Time: O(2^n × n), Space: O(1) excluding output
    """
    n = len(nums)
    result = []
    
    # Generate all 2^n combinations
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            # Check if ith bit is set
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result

# Example usage
print(subsets_iterative([1, 2, 3]))
```

## Pattern 4: Palindrome Partitioning

### Example 1: Partition into Palindromes

**Problem:** Partition string into all possible palindromic substrings.

```python
def partition(s):
    """
    Time: O(n × 2^n), Space: O(n)
    """
    result = []
    
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path.copy())
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            
            if is_palindrome(substring):
                path.append(substring)
                backtrack(end, path)
                path.pop()
    
    backtrack(0, [])
    return result

# Example usage
print(partition("aab"))
# [["a","a","b"], ["aa","b"]]
```

### Example 2: Palindrome Partitioning II (Minimum Cuts)

**Problem:** Find minimum cuts needed for palindrome partitioning.

```python
def minCut(s):
    """
    Time: O(n²), Space: O(n²)
    Using DP + backtracking concept
    """
    n = len(s)
    
    # Build palindrome table
    is_palin = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palin[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_palin[i][j] = (length == 2) or is_palin[i+1][j-1]
    
    # DP for minimum cuts
    dp = [float('inf')] * n
    
    for i in range(n):
        if is_palin[0][i]:
            dp[i] = 0
        else:
            for j in range(i):
                if is_palin[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n-1]

# Example usage
print(minCut("aab"))   # 1
print(minCut("ababbbabbababa"))  # 3
```

## Pattern 5: Word Search & Letter Combinations

### Example 1: Word Search

**Problem:** Find if word exists in 2D board (can move up/down/left/right).

```python
def exist(board, word):
    """
    Time: O(m × n × 4^L), Space: O(L)
    where L is length of word
    """
    rows, cols = len(board), len(board[0])
    
    def backtrack(r, c, index):
        # Found complete word
        if index == len(word):
            return True
        
        # Out of bounds or character mismatch
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[index]):
            return False
        
        # Mark as visited
        temp = board[r][c]
        board[r][c] = '#'
        
        # Explore all 4 directions
        found = (backtrack(r + 1, c, index + 1) or
                backtrack(r - 1, c, index + 1) or
                backtrack(r, c + 1, index + 1) or
                backtrack(r, c - 1, index + 1))
        
        # Restore
        board[r][c] = temp
        
        return found
    
    # Try starting from each cell
    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    
    return False

# Example usage
board = [
    ['A','B','C','E'],
    ['S','F','C','S'],
    ['A','D','E','E']
]
print(exist(board, "ABCCED"))  # True
print(exist(board, "SEE"))     # True
print(exist(board, "ABCB"))    # False
```

### Example 2: Letter Combinations of Phone Number

**Problem:** Generate all letter combinations that number could represent.

```python
def letterCombinations(digits):
    """
    Time: O(4^n × n), Space: O(n)
    """
    if not digits:
        return []
    
    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, path):
        if index == len(digits):
            result.append(''.join(path))
            return
        
        for letter in phone[digits[index]]:
            path.append(letter)
            backtrack(index + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# Example usage
print(letterCombinations("23"))
# ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

### Example 3: Generate Parentheses

**Problem:** Generate all valid combinations of n pairs of parentheses.

```python
def generateParenthesis(n):
    """
    Time: O(4^n / sqrt(n)), Space: O(n)
    """
    result = []
    
    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        
        # Add opening parenthesis
        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()
        
        # Add closing parenthesis (only if valid)
        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()
    
    backtrack([], 0, 0)
    return result

# Example usage
print(generateParenthesis(3))
# ["((()))","(()())","(())()","()(())","()()()"]
```

## Pattern 6: N-Queens Problem

### Example: N-Queens

**Problem:** Place n queens on n×n board such that no two queens attack each other.

```python
def solveNQueens(n):
    """
    Time: O(n!), Space: O(n²)
    """
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def is_valid(row, col):
        # Check column
        for r in range(row):
            if board[r][col] == 'Q':
                return False
        
        # Check upper left diagonal
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if board[r][c] == 'Q':
                return False
            r -= 1
            c -= 1
        
        # Check upper right diagonal
        r, c = row - 1, col + 1
        while r >= 0 and c < n:
            if board[r][c] == 'Q':
                return False
            r -= 1
            c += 1
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_valid(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    backtrack(0)
    return result

# Optimized with sets
def solveNQueens_optimized(n):
    """
    Time: O(n!), Space: O(n)
    Using sets for O(1) validation
    """
    result = []
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col
    board = [['.'] * n for _ in range(n)]
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            backtrack(row + 1)
            
            # Remove queen
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    backtrack(0)
    return result

# Example usage
print(len(solveNQueens(4)))  # 2 solutions
```

## Pattern 7: Sudoku Solver

### Example: Solve Sudoku

**Problem:** Fill a 9×9 Sudoku board following the rules.

```python
def solveSudoku(board):
    """
    Time: O(9^m) where m is empty cells, Space: O(1)
    """
    def is_valid(row, col, num):
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        for r in range(9):
            if board[r][col] == num:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False
        
        return True
    
    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num
                            
                            if backtrack():
                                return True
                            
                            board[row][col] = '.'
                    
                    return False  # No valid number found
        
        return True  # Board filled successfully
    
    backtrack()

# Example usage
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
solveSudoku(board)
```

## Common Pitfalls & Tips

### Pitfalls to Avoid:
- **Forgetting to backtrack** - Always undo choices
- **Not using `path.copy()`** - Appending reference instead of copy
- **Wrong start index** - Use `i` for reuse, `i+1` for no reuse
- **Missing duplicate handling** - Sort and skip duplicates at same level
- **Inefficient validation** - Use sets for O(1) checks
- **Not pruning early** - Check constraints before recursing
- **Modifying input** - Restore original state after backtrack

### Pro Tips:
- **Always copy** when adding to result: `result.append(path.copy())`
- **Sort for duplicates** - Makes skipping easier
- **Use start index** - Prevents generating same combination multiple times
- **Prune early** - Check constraints before making recursive call
- **Track visited** - Use sets/booleans for O(1) lookup
- **Optimize validation** - Pre-compute or use mathematical properties
- **Draw decision tree** - Helps visualize backtracking process

## Complexity Analysis

### Time Complexity:
| Problem Type | Time Complexity |
|--------------|-----------------|
| Permutations | O(n! × n) |
| Combinations | O(C(n,k) × k) |
| Subsets | O(2^n × n) |
| N-Queens | O(n!) |
| Sudoku | O(9^m) where m = empty cells |
| Word Search | O(m × n × 4^L) |

### Space Complexity:
- **Recursion stack:** O(n) or O(height of decision tree)
- **Path storage:** O(n)
- **Result storage:** Varies by problem (factorial to exponential)

## When to Use Backtracking

**Use Backtracking when:**
- ✅ Need to find ALL solutions
- ✅ Generate permutations/combinations
- ✅ Constraint satisfaction problems
- ✅ Decision tree exploration
- ✅ Puzzle solving (Sudoku, N-Queens)
- ✅ Path finding with multiple routes
- ✅ Subset generation

**Don't Use Backtracking when:**
- ❌ Need only ONE optimal solution (use DP/Greedy)
- ❌ Can solve with iteration/simple loops
- ❌ Problem has optimal substructure (use DP)
- ❌ Counting solutions without generating (use math/DP)

## Problem Categories for Practice

### 1. Permutations
- Permutations I, II
- Next permutation
- Permutation sequence
- String permutation

### 2. Combinations
- Combinations
- Combination sum I, II, III, IV
- Letter combinations
- Factor combinations

### 3. Subsets
- Subsets I, II
- Increasing subsequences
- Iterator for combination

### 4. Partition Problems
- Palindrome partitioning I, II
- Partition to K equal sum subsets
- Partition equal subset sum

### 5. Board/Grid Problems
- N-Queens I, II
- Sudoku solver
- Word search I, II
- Unique paths III

### 6. String Problems
- Letter case permutation
- Remove invalid parentheses
- Generate parentheses
- Restore IP addresses

### 7. Advanced
- Expression add operators
- Concatenated words
- Word squares
- Matchsticks to square

## Quick Reference

### Permutation Template
```python
def permute(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path.copy())
            return
        for i in range(len(remaining)):
            backtrack(path + [remaining[i]], 
                     remaining[:i] + remaining[i+1:])
    backtrack([], nums)
    return result
```

### Combination Template
```python
def combine(n, k):
    result = []
    def backtrack(start, path):
        if len(path) == k:
            result.append(path.copy())
            return
        for i in range(start, n + 1):
            backtrack(i + 1, path + [i])
    backtrack(1, [])
    return result
```

### Subset Template
```python
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path.copy())
        for i in range(start, len(nums)):
            backtrack(i + 1, path + [nums[i]])
    backtrack(0, [])
    return result
```


# Backtracking Algorithm Guide

## Key Characteristics:
* **Try all possibilities** - Exhaustive search
* **Choose, Explore, Unchoose** - Core backtracking pattern
* **Pruning** - Skip invalid paths early
* **Recursive nature** - Natural fit for decision trees

## Common Patterns:
1. **Permutations** - n! arrangements, all elements used
2. **Combinations** - Choose k from n, order doesn't matter
3. **Subsets** - All possible selections (power set)
4. **Partition** - Split into valid groups
5. **Board games** - N-Queens, Sudoku
6. **Path finding** - Word search, maze solving

## Key Differences:

| Type | Formula | Reuse? | Order? | Start Index |
|------|---------|--------|--------|-------------|
| Permutation | n! | No | Yes | Any |
| Combination | C(n,k) | No | No | i+1 |
| Subset | 2^n | No | No | i+1 |
| Comb w/ Reuse | - | Yes | No | i |

## Optimization Techniques:
1. **Sort first** - Helps skip duplicates
2. **Use sets** - O(1) validation checks
3. **Prune early** - Check constraints before recursing
4. **Memoization** - Cache results for overlapping subproblems
5. **Bit manipulation** - For subsets generation
6. **Mathematical properties** - Like diagonals in N-Queens

## Key Takeaway:
Backtracking is about systematically trying all possibilities while intelligently pruning invalid paths. Master the three templates (permutations, combinations, subsets) and you can solve most backtracking problems by adapting these patterns!

Practice drawing decision trees to understand the backtracking process. Remember: **Choose → Explore → Unchoose!**
