# Prefix Sum Algorithm: Complete Guide

## What is Prefix Sum?

Prefix Sum (also called Cumulative Sum) is a preprocessing technique where we create an auxiliary array that stores the sum of elements from the start up to each index. This allows us to answer range sum queries in constant time O(1) after O(n) preprocessing.

Think of it like a running total on your bank statement - each entry shows the cumulative balance up to that point, making it easy to calculate the difference between any two dates.

## Core Strategy

The key insight: **Precompute cumulative sums once, then answer range queries instantly.**

Instead of calculating sum repeatedly:
```
sum(arr[i:j]) = arr[i] + arr[i+1] + ... + arr[j-1]  # O(n) for each query
```

We use prefix sum:
```
prefix[i] = arr[0] + arr[1] + ... + arr[i-1]
sum(arr[i:j]) = prefix[j] - prefix[i]  # O(1) for each query
```

## Basic Concept

### Building Prefix Sum Array

Given array: `[3, 1, 4, 1, 5, 9, 2]`

Prefix sum array:
```
Index:      0   1   2   3   4   5   6   7
Original:   -   3   1   4   1   5   9   2
Prefix:     0   3   4   8   9  14  23  25
            ↑
      (0 for convenience)
```

**Formula:** `prefix[i] = prefix[i-1] + arr[i-1]`

### Range Sum Query

To find sum of elements from index `i` to `j` (inclusive):

```
sum(i, j) = prefix[j+1] - prefix[i]
```

**Example:** Sum from index 2 to 5
```
sum(2, 5) = prefix[6] - prefix[2]
          = 23 - 4
          = 19
          = 4 + 1 + 5 + 9 ✓
```

## Types of Prefix Sum Problems

### 1. Basic Range Sum Queries
Calculate sum of elements in a given range multiple times.

### 2. Subarray Sum Problems
Find subarrays with specific sum properties.

### 3. 2D Prefix Sum
Handle range sum queries in matrices.

### 4. XOR/Product Prefix
Apply prefix concept to other operations.

### 5. Difference Array (Inverse)
Efficiently apply range updates.

## Step-by-Step Problem-Solving Framework

### Step 1: Identify if Prefix Sum Applies

Ask yourself:
- Do I need to calculate **sum of ranges** multiple times?
- Am I looking for **subarrays with specific sum**?
- Can I transform the problem into a **cumulative computation**?
- Would computing sums repeatedly be inefficient?

**Keywords in problems:** "range sum", "subarray sum", "cumulative", "running total", "sum equals k"

### Step 2: Choose the Right Variant

**Choose Basic Prefix Sum if:**
- Need to answer multiple range sum queries
- Array doesn't change (or changes are rare)
- Example: "Find sum of elements from index L to R"

**Choose with HashMap if:**
- Finding subarrays with target sum
- Counting subarrays with properties
- Example: "Count subarrays with sum equals K"

**Choose 2D Prefix Sum if:**
- Working with matrices
- Need rectangular region sums
- Example: "Sum of rectangle in matrix"

**Choose Difference Array if:**
- Multiple range updates needed
- Few queries after all updates
- Example: "Add value to range [L, R] multiple times"

### Step 3: Build Prefix Sum Array

```python
# For 1D array
prefix = [0]
for num in arr:
    prefix.append(prefix[-1] + num)

# Or more concisely
prefix = [0] + list(accumulate(arr))
```

### Step 4: Use Prefix Sum for Queries/Solutions

Apply the appropriate formula based on problem type.

## Pattern 1: Basic Range Sum Queries

### Template
```python
class RangeSumQuery:
    def __init__(self, nums):
        """
        Initialize with O(n) preprocessing
        """
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
    
    def sum_range(self, left, right):
        """
        Query in O(1) time
        """
        return self.prefix[right + 1] - self.prefix[left]
```

### Example 1: Range Sum Query - Immutable

**Problem:** Given an array, answer multiple range sum queries efficiently.

**Solution Steps:**
1. Build prefix sum array in constructor
2. Use prefix difference formula for queries

```python
class NumArray:
    """
    Preprocessing: O(n), Query: O(1), Space: O(n)
    """
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
    
    def sumRange(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]

# Example usage
nums = [-2, 0, 3, -5, 2, -1]
obj = NumArray(nums)
print(obj.sumRange(0, 2))  # Output: 1 (-2 + 0 + 3)
print(obj.sumRange(2, 5))  # Output: -1 (3 + -5 + 2 + -1)
print(obj.sumRange(0, 5))  # Output: -3
```

### Example 2: Find Pivot Index

**Problem:** Find index where sum of left elements equals sum of right elements.

**Solution Steps:**
1. Calculate total sum
2. Iterate while tracking left sum
3. Check if left sum equals right sum (total - left - current)

```python
def pivot_index(nums):
    """
    Time: O(n), Space: O(1)
    """
    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        # Right sum = total - left - current
        right_sum = total_sum - left_sum - num
        
        if left_sum == right_sum:
            return i
        
        left_sum += num
    
    return -1

# Example usage
nums = [1, 7, 3, 6, 5, 6]
print(pivot_index(nums))  # Output: 3 (left sum: 1+7+3=11, right sum: 5+6=11)
```

### Example 3: Product of Array Except Self

**Problem:** Return array where each element is product of all others (without division).

**Solution Steps:**
1. Build prefix product from left
2. Build suffix product from right
3. Multiply left and right products

```python
def product_except_self(nums):
    """
    Time: O(n), Space: O(1) excluding output
    """
    n = len(nums)
    result = [1] * n
    
    # Left products
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Right products
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result

# Example usage
nums = [1, 2, 3, 4]
print(product_except_self(nums))  # Output: [24, 12, 8, 6]
```

## Pattern 2: Subarray Sum with HashMap

### Template
```python
def subarray_sum_with_target(arr, k):
    prefix_sum = 0
    count = 0
    sum_freq = {0: 1}  # prefix_sum: frequency
    
    for num in arr:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if (prefix_sum - k) in sum_freq:
            count += sum_freq[prefix_sum - k]
        
        # Update frequency
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count
```

### Example 1: Subarray Sum Equals K

**Problem:** Count number of continuous subarrays that sum to k.

**Key Insight:** If `prefix[j] - prefix[i] = k`, then `prefix[i] = prefix[j] - k`

**Solution Steps:**
1. Track prefix sums in a hashmap with their frequencies
2. For each position, check if `(current_prefix - k)` exists
3. If yes, we found subarrays ending at current position

```python
def subarray_sum(nums, k):
    """
    Time: O(n), Space: O(n)
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # Base case: empty subarray has sum 0
    
    for num in nums:
        prefix_sum += num
        
        # Check how many times (prefix_sum - k) occurred
        if (prefix_sum - k) in sum_freq:
            count += sum_freq[prefix_sum - k]
        
        # Update frequency of current prefix sum
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count

# Example usage
nums = [1, 1, 1]
k = 2
print(subarray_sum(nums, k))  # Output: 2 ([1,1] appears twice)

nums = [1, 2, 3]
k = 3
print(subarray_sum(nums, k))  # Output: 2 ([1,2] and [3])
```

### Example 2: Contiguous Array (0s and 1s)

**Problem:** Find longest subarray with equal number of 0s and 1s.

**Key Insight:** Treat 0 as -1, then problem becomes "longest subarray with sum 0"

**Solution Steps:**
1. Convert 0 to -1
2. Find longest subarray where prefix sum difference is 0
3. Track first occurrence of each prefix sum

```python
def find_max_length(nums):
    """
    Time: O(n), Space: O(n)
    """
    max_length = 0
    prefix_sum = 0
    sum_index = {0: -1}  # prefix_sum: first_index
    
    for i, num in enumerate(nums):
        # Treat 0 as -1
        prefix_sum += 1 if num == 1 else -1
        
        if prefix_sum in sum_index:
            # Found subarray with equal 0s and 1s
            max_length = max(max_length, i - sum_index[prefix_sum])
        else:
            # Store first occurrence
            sum_index[prefix_sum] = i
    
    return max_length

# Example usage
nums = [0, 1, 0]
print(find_max_length(nums))  # Output: 2 ([0,1] or [1,0])

nums = [0, 1, 0, 1]
print(find_max_length(nums))  # Output: 4 (entire array)
```

### Example 3: Subarray Divisible by K

**Problem:** Count subarrays with sum divisible by k.

**Key Insight:** If `(prefix[j] - prefix[i]) % k == 0`, then `prefix[j] % k == prefix[i] % k`

```python
def subarrays_div_by_k(nums, k):
    """
    Time: O(n), Space: O(k)
    """
    count = 0
    prefix_sum = 0
    remainder_freq = {0: 1}  # remainder: frequency
    
    for num in nums:
        prefix_sum += num
        remainder = prefix_sum % k
        
        # Handle negative remainders
        if remainder < 0:
            remainder += k
        
        # Add count of subarrays ending at current position
        count += remainder_freq.get(remainder, 0)
        
        # Update frequency
        remainder_freq[remainder] = remainder_freq.get(remainder, 0) + 1
    
    return count

# Example usage
nums = [4, 5, 0, -2, -3, 1]
k = 5
print(subarrays_div_by_k(nums, k))  # Output: 7
```

## Pattern 3: 2D Prefix Sum (Matrix)

### Concept

For a matrix, 2D prefix sum at (i, j) represents sum of all elements in rectangle from (0, 0) to (i, j).

**Formula:**
```
prefix[i][j] = prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1] + matrix[i-1][j-1]
```

**Range Query (from (r1,c1) to (r2,c2)):**
```
sum = prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
```

### Example: Range Sum Query 2D

**Problem:** Calculate sum of elements in any rectangular region efficiently.

```python
class NumMatrix:
    """
    Preprocessing: O(m*n), Query: O(1), Space: O(m*n)
    """
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        m, n = len(matrix), len(matrix[0])
        # Create prefix sum matrix with extra row and column
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self.prefix[i][j] = (
                    self.prefix[i-1][j] +      # Top
                    self.prefix[i][j-1] -      # Left
                    self.prefix[i-1][j-1] +    # Top-left (subtracted twice)
                    matrix[i-1][j-1]           # Current cell
                )
    
    def sumRegion(self, row1, col1, row2, col2):
        """
        Calculate sum of rectangle from (row1, col1) to (row2, col2)
        """
        return (
            self.prefix[row2+1][col2+1] -
            self.prefix[row1][col2+1] -
            self.prefix[row2+1][col1] +
            self.prefix[row1][col1]
        )

# Example usage
matrix = [
    [3, 0, 1, 4, 2],
    [5, 6, 3, 2, 1],
    [1, 2, 0, 1, 5],
    [4, 1, 0, 1, 7],
    [1, 0, 3, 0, 5]
]
obj = NumMatrix(matrix)
print(obj.sumRegion(2, 1, 4, 3))  # Output: 8
print(obj.sumRegion(1, 1, 2, 2))  # Output: 11
```

### Visualization of 2D Prefix Sum

```
Original Matrix:        Prefix Sum Matrix:
1  2  3                 0  0  0  0
4  5  6        →        0  1  3  6
7  8  9                 0  5  12 21
                        0  12 27 45

Query sum(1,1 to 2,2):
= prefix[3][3] - prefix[1][3] - prefix[3][1] + prefix[1][1]
= 45 - 6 - 12 + 1 = 28
= 5 + 6 + 8 + 9 = 28 ✓
```

## Pattern 4: Difference Array (Range Updates)

### Concept

Difference array is the inverse of prefix sum - it efficiently handles range updates.

**Use Case:** When you need to add a value to a range [L, R] multiple times, then query final array.

### Template

```python
def range_updates(n, updates):
    """
    updates: list of (start, end, value) tuples
    Time: O(n + u) where u is number of updates
    """
    diff = [0] * (n + 1)
    
    # Apply updates to difference array
    for start, end, value in updates:
        diff[start] += value
        diff[end + 1] -= value
    
    # Convert back to original array using prefix sum
    result = [0] * n
    result[0] = diff[0]
    for i in range(1, n):
        result[i] = result[i-1] + diff[i]
    
    return result
```

### Example: Range Addition

**Problem:** Start with array of zeros. Apply multiple range additions. Return final array.

```python
def get_modified_array(length, updates):
    """
    Time: O(n + k) where k is number of updates
    Space: O(n)
    """
    diff = [0] * (length + 1)
    
    # Apply each update to difference array
    for start, end, inc in updates:
        diff[start] += inc
        diff[end + 1] -= inc
    
    # Build result using prefix sum
    result = []
    current = 0
    for i in range(length):
        current += diff[i]
        result.append(current)
    
    return result

# Example usage
length = 5
updates = [[1, 3, 2], [2, 4, 3], [0, 2, -2]]
print(get_modified_array(length, updates))
# Output: [-2, 0, 3, 5, 3]
```

### How Difference Array Works

```
Array length: 5 (indices 0-4)
Initial: [0, 0, 0, 0, 0]

Update 1: Add 2 to range [1, 3]
diff: [0, 2, 0, 0, -2, 0]

Update 2: Add 3 to range [2, 4]
diff: [0, 2, 3, 0, -2, -3]

Update 3: Add -2 to range [0, 2]
diff: [-2, 2, 3, -2, -2, -3]

Apply prefix sum to diff:
[-2, 0, 3, 5, 3]
```

## Pattern 5: XOR Prefix

### Concept

Similar to sum, but using XOR operation. Useful for problems involving XOR properties.

**Property:** `A ⊕ A = 0` and `A ⊕ 0 = A`

### Example: XOR Queries of a Subarray

```python
class XorQuery:
    """
    Preprocessing: O(n), Query: O(1)
    """
    def __init__(self, arr):
        self.xor_prefix = [0]
        for num in arr:
            self.xor_prefix.append(self.xor_prefix[-1] ^ num)
    
    def query(self, left, right):
        """
        XOR of elements from index left to right
        """
        return self.xor_prefix[right + 1] ^ self.xor_prefix[left]

# Example usage
arr = [1, 3, 4, 8]
xor_query = XorQuery(arr)
print(xor_query.query(0, 1))  # 1 ^ 3 = 2
print(xor_query.query(1, 3))  # 3 ^ 4 ^ 8 = 15
```

## Advanced Problems

### Example 1: Maximum Subarray Sum (Kadane's Algorithm)

**Problem:** Find contiguous subarray with largest sum.

**Prefix Sum Perspective:**
```python
def max_subarray(nums):
    """
    Time: O(n), Space: O(1)
    """
    max_sum = float('-inf')
    current_sum = 0
    min_prefix = 0
    
    for num in nums:
        current_sum += num
        max_sum = max(max_sum, current_sum - min_prefix)
        min_prefix = min(min_prefix, current_sum)
    
    return max_sum

# Traditional Kadane's
def max_subarray_kadane(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

### Example 2: Count Subarrays with Bounded Maximum

**Problem:** Count subarrays where maximum element is in range [L, R].

```python
def num_subarray_bounded_max(nums, left, right):
    """
    Time: O(n), Space: O(1)
    """
    def count_with_max_at_most(bound):
        count = 0
        current = 0
        for num in nums:
            if num <= bound:
                current += 1
                count += current
            else:
                current = 0
        return count
    
    # Subarrays with max in [L,R] = 
    # (max <= R) - (max <= L-1)
    return count_with_max_at_most(right) - count_with_max_at_most(left - 1)
```

## Common Pitfalls & Tips

### Pitfalls to Avoid:
- **Off-by-one errors** in range queries (inclusive vs exclusive)
- **Forgetting base case** - prefix[0] = 0
- **Not handling negative numbers** properly in modulo operations
- **Wrong formula** for 2D queries (forgetting to add back overlap)
- **Space for difference array** - need n+1 size for proper updates

### Pro Tips:
- **Always add prefix[0] = 0** - makes formulas cleaner
- **Draw it out** - Visualize the prefix sum array
- **Use HashMap** for subarray sum problems with targets
- **Handle negative remainders** - add k if remainder < 0
- **1-indexed can be easier** for 2D prefix sum
- **Test edge cases** - Empty array, single element, all negatives
- **Difference array** is inverse operation - converts prefix sum back

## Complexity Analysis

### Time Complexity:
- **Preprocessing:** O(n) for 1D, O(m×n) for 2D
- **Query:** O(1) for both 1D and 2D
- **With HashMap:** O(n) for single pass problems
- **Difference Array:** O(n + k) where k is updates

### Space Complexity:
- **Basic:** O(n) for 1D, O(m×n) for 2D
- **With HashMap:** O(n) in worst case
- **Optimized:** O(1) when using running sum instead of array

## When to Use Prefix Sum

**Use Prefix Sum when:**
- ✅ Multiple range sum queries on static array
- ✅ Finding subarrays with specific sum properties
- ✅ Need O(1) query time after preprocessing
- ✅ Working with cumulative operations (sum, XOR, product)
- ✅ Problems involving "continuous subarray"
- ✅ 2D matrix range queries

**Don't Use Prefix Sum when:**
- ❌ Array is frequently modified (consider segment tree)
- ❌ Need maximum/minimum in range (use sparse table or segment tree)
- ❌ Elements can be non-contiguous
- ❌ Single query on large array (just iterate)

## Problem Categories for Practice

### 1. Basic Range Queries
- Range sum query - immutable
- Find pivot index
- Running sum of 1D array
- Product of array except self

### 2. Subarray Sum with Target
- Subarray sum equals K
- Continuous subarray sum
- Subarray sum divisible by K
- Maximum size subarray sum equals K

### 3. Subarray Properties
- Contiguous array (equal 0s and 1s)
- Longest well-performing interval
- Count number of nice subarrays
- Binary subarrays with sum

### 4. 2D Problems
- Range sum query 2D - immutable
- Number of submatrices that sum to target
- Max sum of rectangle no larger than K

### 5. Difference Array
- Range addition
- Corporate flight bookings
- Car pooling

### 6. XOR Prefix
- XOR queries of a subarray
- Count triplets with XOR
- Maximum XOR of two numbers

## Comparison with Other Techniques

| Technique | Queries | Updates | Query Time | Space |
|-----------|---------|---------|------------|-------|
| **Prefix Sum** | Multiple | None/Rare | O(1) | O(n) |
| **Difference Array** | Few | Multiple | O(n) | O(n) |
| **Segment Tree** | Multiple | Multiple | O(log n) | O(n) |
| **Fenwick Tree** | Multiple | Multiple | O(log n) | O(n) |
| **Naive** | Any | Any | O(n) | O(1) |

## Quick Reference

### 1D Prefix Sum Formula
```python
# Build
prefix[i] = prefix[i-1] + arr[i-1]

# Query sum from i to j (inclusive)
sum = prefix[j+1] - prefix[i]
```

### 2D Prefix Sum Formula
```python
# Build
prefix[i][j] = prefix[i-1][j] + prefix[i][j-1] 
               - prefix[i-1][j-1] + matrix[i-1][j-1]

# Query rectangle (r1,c1) to (r2,c2)
sum = prefix[r2+1][c2+1] - prefix[r1][c2+1] 
      - prefix[r2+1][c1] + prefix[r1][c1]
```

### Subarray Sum with HashMap
```python
# Pattern
sum_freq = {0: 1}
for num in arr:
    prefix_sum += num
    count += sum_freq.get(prefix_sum - target, 0)
    sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
```

## Summary

Prefix Sum is a powerful preprocessing technique that enables:
- **O(1) range queries** after O(n) preprocessing
- **Efficient subarray sum problems** using HashMap
- **2D matrix queries** with clever formula application
- **Range updates** through difference array (inverse operation)

**Key Takeaway:** Whenever you see repeated range sum queries or subarray sum problems, think prefix sum! Precompute once, query many times efficiently.

Master these patterns and you'll solve range query problems with elegance and optimal time complexity!
