# Sliding Window Algorithm: Complete Guide

## What is Sliding Window?

The sliding window algorithm is a technique that maintains a subset of data (a "window") that moves through a larger dataset. Instead of recalculating everything from scratch for each position, we efficiently update our calculations as the window slides.

Think of it like looking through a camera viewfinder as you pan across a landscape - you're always viewing a fixed or variable portion, smoothly transitioning what you see.

## Core Strategy

The key insight: **Avoid redundant work by reusing calculations from the previous window position.**

Instead of:
- Calculate for positions [0,2]
- Calculate for positions [1,3]  
- Calculate for positions [2,4]

We do:
- Calculate for positions [0,2]
- Remove element at 0, add element at 3
- Remove element at 1, add element at 4

## Types of Sliding Windows

### 1. Fixed-Size Window
The window size remains constant throughout.

**Example Use Cases:**
- Maximum sum of k consecutive elements
- Average of k elements at each position
- Finding anagrams of a pattern in a string

### 2. Variable-Size Window
The window expands and contracts based on conditions.

**Example Use Cases:**
- Longest substring without repeating characters
- Minimum window substring containing all characters
- Longest subarray with sum ≤ k

## Step-by-Step Problem-Solving Framework

### Step 1: Identify if Sliding Window Applies

Ask yourself:
- Am I looking for something in **contiguous** subarrays/substrings?
- Can I build a solution incrementally by adding/removing elements?
- Is there a way to track state efficiently as I move through the data?

**Keywords in problems:** "consecutive", "contiguous", "subarray", "substring", "window"

### Step 2: Choose Fixed or Variable Window

**Choose Fixed if:**
- The problem explicitly mentions a size (k elements, k days, etc.)
- Example: "maximum sum of any subarray of size k"

**Choose Variable if:**
- You need to find the optimal window size
- Conditions determine when to expand/shrink
- Example: "longest substring with at most 2 distinct characters"

### Step 3: Define Your Window State

What do you need to track?
- Sum of elements (for sum problems)
- Frequency map (for character/element counting)
- Min/Max values (for range problems)
- Set (for uniqueness checks)

### Step 4: Implement the Pattern

## Fixed Window Pattern

```
1. Calculate initial window (first k elements)
2. For each new position:
   a. Remove leftmost element from window
   b. Add new rightmost element
   c. Update answer if needed
```

**Code Template:**
```python
def fixed_window(arr, k):
    # Step 1: Initial window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Step 2: Slide the window
    for i in range(k, len(arr)):
        # Remove left element, add right element
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

## Variable Window Pattern

```
1. Use two pointers (left and right)
2. Expand window (move right):
   - Add element to window
   - Update state
3. Contract window (move left) while condition violated:
   - Remove element from window
   - Update state
4. Update answer when condition satisfied
```

**Code Template:**
```python
def variable_window(arr):
    left = 0
    window_state = {}  # or sum, set, etc.
    result = 0
    
    for right in range(len(arr)):
        # Expand: Add arr[right] to window
        # Update window_state
        
        while condition_violated:
            # Contract: Remove arr[left] from window
            # Update window_state
            left += 1
        
        # Update result with valid window
        result = max(result, right - left + 1)
    
    return result
```

## Real Problem Examples

### Example 1: Maximum Sum Subarray of Size K (Fixed)

**Problem:** Find maximum sum of k consecutive elements.

**Solution Steps:**
1. Calculate sum of first k elements
2. Slide window: subtract arr[i-k], add arr[i]
3. Track maximum

```python
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return -1
    
    current_sum = sum(arr[:k])
    max_sum = current_sum
    
    for i in range(k, len(arr)):
        current_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

**Time:** O(n), **Space:** O(1)

### Example 2: Longest Substring Without Repeating Characters (Variable)

**Problem:** Find length of longest substring with unique characters.

**Solution Steps:**
1. Expand window by moving right pointer
2. Track characters in a set/map
3. When duplicate found, shrink from left
4. Track maximum length

```python
def longest_unique_substring(s):
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add current character
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

**Time:** O(n), **Space:** O(min(n, alphabet_size))

### Example 3: Minimum Window Substring (Variable - Advanced)

**Problem:** Find minimum window in string S that contains all characters from string T.

**Solution Steps:**
1. Use frequency map for target characters
2. Expand window until all characters found
3. Contract window while maintaining validity
4. Track minimum window

```python
def min_window_substring(s, t):
    from collections import Counter
    
    if not s or not t:
        return ""
    
    target_count = Counter(t)
    required = len(target_count)
    formed = 0
    
    window_counts = {}
    left = 0
    min_len = float('inf')
    min_window = (0, 0)
    
    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in target_count and window_counts[char] == target_count[char]:
            formed += 1
        
        while left <= right and formed == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_window = (left, right)
            
            char = s[left]
            window_counts[char] -= 1
            if char in target_count and window_counts[char] < target_count[char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_window[0]:min_window[1] + 1]
```

**Time:** O(|S| + |T|), **Space:** O(|S| + |T|)

## Common Pitfalls & Tips

### Pitfalls to Avoid:
- Forgetting to handle edge cases (empty input, k > length)
- Off-by-one errors in window boundaries
- Not properly updating state when shrinking window
- Using wrong data structure for tracking state
- Infinite loops when condition for shrinking window is incorrect

### Pro Tips:
- Draw out the window movement on paper first
- Verify your window size calculation: `right - left + 1`
- For character problems, use arrays of size 128/256 instead of hashmaps for better performance
- When stuck, ask: "What information do I lose when sliding left? What do I gain when sliding right?"
- Test with edge cases: empty array, single element, all same elements, k = 1, k = array length

## Complexity Analysis

**Time Complexity:** Typically O(n)
- Each element is visited at most twice (once by right pointer, once by left pointer)
- Fixed window: exactly O(n)
- Variable window: O(n) amortized

**Space Complexity:** Varies
- O(1) for simple sum/count tracking
- O(k) for tracking k elements in window
- O(alphabet_size) for character frequency problems (often O(1) since alphabet is constant)

## Practice Problem Categories

### 1. Fixed Window Problems
- Maximum/minimum sum of k elements
- Average of subarrays of size k
- First negative number in every window of size k
- Count occurrences of anagrams

### 2. Variable Window with Frequency
- Longest substring with k distinct characters
- Find all anagrams in a string
- Permutation in string
- Substring with concatenation of all words

### 3. Variable Window with Constraints
- Longest substring without repeating characters
- Longest repeating character replacement
- Subarray sum equals k
- Maximum consecutive ones (with k flips allowed)

### 4. Two Pointers Variant
- Container with most water
- Trapping rain water
- 3Sum, 4Sum problems

## Common Problem Patterns

### Pattern 1: Max/Min in Fixed Window
Use case: "Find maximum sum of k consecutive elements"
```python
# Track: sum, max, min, product, etc.
# Template: calculate initial, then slide
```

### Pattern 2: Count Occurrences
Use case: "Count anagrams of pattern in string"
```python
# Track: frequency map
# Template: match frequency, slide, compare
```

### Pattern 3: Longest/Shortest with Condition
Use case: "Longest substring with at most k distinct characters"
```python
# Track: frequency map, distinct count
# Template: expand right, shrink left when invalid
```

### Pattern 4: Find All Windows
Use case: "Find all substrings that are anagrams"
```python
# Track: frequency map, matches
# Template: fixed window, check each position
```

## When NOT to Use Sliding Window

Sliding window may not be appropriate when:
- Elements don't need to be contiguous
- Order doesn't matter (might need sorting instead)
- You need to find subarrays with specific properties that can't be tracked incrementally
- Problem requires checking all possible subsets (might need backtracking)
- Looking for specific indices or positions rather than values

## Additional Resources

**Similar Techniques:**
- Two Pointers: Often used together with sliding window
- Monotonic Queue: For min/max in window problems
- Prefix Sum: Alternative for some fixed window problems

**Practice Platforms:**
- LeetCode: Tag "Sliding Window"
- HackerRank: Array manipulation problems
- Codeforces: Substring problems

## Summary

The sliding window technique is powerful because it transforms brute force O(n²) or O(n·k) solutions into elegant O(n) algorithms by intelligently reusing computations. Master the two patterns (fixed and variable) and you'll be able to recognize and solve a wide variety of problems efficiently.

**Key Takeaway:** Always ask yourself - "Can I avoid recalculating everything by just updating what changed?" If yes, sliding window might be your answer!
