# Two Pointers Algorithm: Complete Guide

## What is Two Pointers?

Two Pointers is a technique where we use two references (pointers) to traverse through a data structure, typically an array or string. The pointers move according to certain conditions, helping us solve problems efficiently without nested loops.

Think of it like two people walking through a line - they might start at opposite ends and move toward each other, or both start at the beginning and move at different speeds, or one chases the other.

## Core Strategy

The key insight: **Use multiple pointers to avoid nested iterations and reduce time complexity from O(n²) to O(n).**

Instead of checking every pair with nested loops:
```
for i in range(n):
    for j in range(i+1, n):  # O(n²)
```

We intelligently move two pointers based on conditions:
```
left, right = 0, n-1
while left < right:  # O(n)
    # make decision and move pointers
```

## Types of Two Pointer Patterns

### 1. Opposite Directional (Converging Pointers)
Pointers start at opposite ends and move toward each other.

**Use Cases:**
- Finding pairs with a target sum in sorted array
- Palindrome checking
- Container with most water
- Reversing arrays/strings

### 2. Same Directional (Fast & Slow Pointers)
Both pointers move in the same direction at different speeds.

**Use Cases:**
- Removing duplicates from sorted array
- Moving zeros to end
- Cycle detection in linked list
- Finding middle of linked list

### 3. Sliding Window Variant
Two pointers maintain a window, both move forward (covered in sliding window guide).

**Use Cases:**
- Longest substring problems
- Subarray sum problems

### 4. Partitioning
Using pointers to partition array into sections.

**Use Cases:**
- Dutch National Flag problem
- Quick sort partition
- Segregating elements

## Step-by-Step Problem-Solving Framework

### Step 1: Identify if Two Pointers Applies

Ask yourself:
- Am I searching for **pairs, triplets**, or **subarrays**?
- Is the array/list **sorted** or can I sort it?
- Do I need to **compare elements** at different positions?
- Can I **eliminate possibilities** by moving pointers?

**Keywords in problems:** "pair", "two elements", "sorted array", "in-place", "partition", "remove duplicates"

### Step 2: Choose the Right Pattern

**Choose Opposite Directional if:**
- Array is sorted
- Looking for pairs/combinations
- Need to check from both ends
- Example: "Find two numbers that sum to target"

**Choose Same Directional if:**
- Need to process elements sequentially
- One pointer explores, other tracks position
- In-place array modifications
- Example: "Remove duplicates from sorted array"

**Choose Partitioning if:**
- Need to rearrange elements
- Segregate based on condition
- Example: "Move all zeros to end"

### Step 3: Define Pointer Movement Logic

Ask:
- When do I move left pointer?
- When do I move right pointer?
- When do I move both?
- What's my termination condition?

### Step 4: Handle Edge Cases

- Empty array/string
- Single element
- All elements same
- No valid answer exists

## Pattern 1: Opposite Directional Pointers

### Template
```python
def opposite_pointers(arr):
    left = 0
    right = len(arr) - 1
    
    while left < right:
        # Check condition
        if condition_met:
            # Process and possibly return
            return result
        elif need_larger_value:
            left += 1  # Move left pointer right
        else:
            right -= 1  # Move right pointer left
    
    return default_result
```

### Example 1: Two Sum in Sorted Array

**Problem:** Find two numbers that add up to a target in a sorted array.

**Solution Steps:**
1. Start with pointers at both ends
2. Calculate sum of elements at pointers
3. If sum equals target, return indices
4. If sum too small, move left pointer right (increase sum)
5. If sum too large, move right pointer left (decrease sum)

```python
def two_sum_sorted(arr, target):
    """
    Time: O(n), Space: O(1)
    """
    left = 0
    right = len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return [-1, -1]  # No solution found

# Example usage
arr = [1, 2, 3, 4, 6, 8, 9]
target = 10
print(two_sum_sorted(arr, target))  # Output: [2, 5] (3 + 8 = 10)
```

### Example 2: Valid Palindrome

**Problem:** Check if a string is a palindrome (ignoring non-alphanumeric characters).

**Solution Steps:**
1. Use two pointers from both ends
2. Skip non-alphanumeric characters
3. Compare characters (case-insensitive)
4. Move pointers inward

```python
def is_palindrome(s):
    """
    Time: O(n), Space: O(1)
    """
    left = 0
    right = len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

# Example usage
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))  # False
```

### Example 3: Container With Most Water

**Problem:** Find two lines that together with x-axis form a container holding the most water.

**Solution Steps:**
1. Start with widest container (leftmost and rightmost lines)
2. Calculate area: min(height[left], height[right]) × width
3. Move the pointer with shorter height (limiting factor)
4. Track maximum area

```python
def max_area(height):
    """
    Time: O(n), Space: O(1)
    """
    left = 0
    right = len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        current_area = min(height[left], height[right]) * width
        max_water = max(max_water, current_area)
        
        # Move pointer with shorter height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Example usage
heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print(max_area(heights))  # Output: 49
```

## Pattern 2: Same Directional Pointers (Fast & Slow)

### Template
```python
def same_direction(arr):
    slow = 0
    
    for fast in range(len(arr)):
        if condition_met:
            # Process element
            arr[slow] = arr[fast]
            slow += 1
    
    return slow  # or arr[:slow]
```

### Example 1: Remove Duplicates from Sorted Array

**Problem:** Remove duplicates in-place, return length of unique elements.

**Solution Steps:**
1. Slow pointer tracks position for next unique element
2. Fast pointer explores array
3. When fast finds new unique element, place it at slow position
4. Increment slow

```python
def remove_duplicates(nums):
    """
    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0
    
    slow = 1  # Position for next unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow

# Example usage
nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
length = remove_duplicates(nums)
print(length)  # Output: 5
print(nums[:length])  # Output: [0, 1, 2, 3, 4]
```

### Example 2: Move Zeros

**Problem:** Move all zeros to the end while maintaining order of non-zero elements.

**Solution Steps:**
1. Slow pointer tracks position for next non-zero
2. Fast pointer finds non-zero elements
3. Swap elements when fast finds non-zero
4. Increment slow

```python
def move_zeros(nums):
    """
    Time: O(n), Space: O(1)
    """
    slow = 0  # Position for next non-zero
    
    for fast in range(len(nums)):
        if nums[fast] != 0:
            # Swap non-zero element to slow position
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

# Example usage
nums = [0, 1, 0, 3, 12]
move_zeros(nums)
print(nums)  # Output: [1, 3, 12, 0, 0]
```

### Example 3: Linked List Cycle Detection (Floyd's Algorithm)

**Problem:** Detect if a linked list has a cycle.

**Solution Steps:**
1. Use slow (moves 1 step) and fast (moves 2 steps) pointers
2. If there's a cycle, fast will eventually meet slow
3. If fast reaches end, no cycle exists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    """
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False
```

### Example 4: Find Middle of Linked List

**Problem:** Find the middle node of a linked list.

**Solution Steps:**
1. Slow moves 1 step, fast moves 2 steps
2. When fast reaches end, slow is at middle
3. For even length, returns second middle node

```python
def find_middle(head):
    """
    Time: O(n), Space: O(1)
    """
    if not head:
        return None
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # Slow is now at middle
```

## Pattern 3: Three Pointers (Extension)

### Example: 3Sum Problem

**Problem:** Find all unique triplets that sum to zero.

**Solution Steps:**
1. Sort the array first
2. Fix one element with outer loop
3. Use two pointers for remaining two elements
4. Skip duplicates to avoid repeated triplets

```python
def three_sum(nums):
    """
    Time: O(n²), Space: O(1) excluding output
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left = i + 1
        right = n - 1
        target = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for second element
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for third element
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result

# Example usage
nums = [-1, 0, 1, 2, -1, -4]
print(three_sum(nums))  # Output: [[-1, -1, 2], [-1, 0, 1]]
```

## Pattern 4: Partitioning

### Example: Dutch National Flag (Sort Colors)

**Problem:** Sort an array containing only 0s, 1s, and 2s in-place.

**Solution Steps:**
1. Use three pointers: low (0s boundary), mid (current), high (2s boundary)
2. Mid explores array
3. Swap elements to appropriate regions
4. Move pointers based on values

```python
def sort_colors(nums):
    """
    Time: O(n), Space: O(1)
    """
    low = 0  # Boundary for 0s
    mid = 0  # Current element
    high = len(nums) - 1  # Boundary for 2s
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # Don't increment mid - need to check swapped element

# Example usage
nums = [2, 0, 2, 1, 1, 0]
sort_colors(nums)
print(nums)  # Output: [0, 0, 1, 1, 2, 2]
```

## Advanced Techniques

### Technique 1: Reverse with Two Pointers

```python
def reverse_array(arr):
    """
    Time: O(n), Space: O(1)
    """
    left = 0
    right = len(arr) - 1
    
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
```

### Technique 2: Squaring Sorted Array

**Problem:** Square elements of sorted array (with negatives) and return sorted result.

```python
def sorted_squares(nums):
    """
    Time: O(n), Space: O(n)
    """
    n = len(nums)
    result = [0] * n
    left = 0
    right = n - 1
    pos = n - 1  # Fill from end
    
    while left <= right:
        left_sq = nums[left] ** 2
        right_sq = nums[right] ** 2
        
        if left_sq > right_sq:
            result[pos] = left_sq
            left += 1
        else:
            result[pos] = right_sq
            right -= 1
        pos -= 1
    
    return result

# Example usage
nums = [-4, -1, 0, 3, 10]
print(sorted_squares(nums))  # Output: [0, 1, 9, 16, 100]
```

### Technique 3: Trapping Rain Water

**Problem:** Calculate how much water can be trapped after raining.

```python
def trap_rain_water(height):
    """
    Time: O(n), Space: O(1)
    """
    if not height:
        return 0
    
    left = 0
    right = len(height) - 1
    left_max = height[left]
    right_max = height[right]
    water = 0
    
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    
    return water

# Example usage
height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
print(trap_rain_water(height))  # Output: 6
```

## Common Pitfalls & Tips

### Pitfalls to Avoid:
- Not handling empty arrays or single elements
- Infinite loops when pointer movement conditions are wrong
- Off-by-one errors in loop conditions (`<` vs `<=`)
- Forgetting to skip duplicates in sum problems
- Not considering negative numbers in sorted array problems
- Moving both pointers when only one should move

### Pro Tips:
- **Always draw it out** - Visualize pointer movements on paper
- **Check boundary conditions** - What happens at edges?
- **Sort when needed** - Many two-pointer problems require sorted input
- **Skip duplicates carefully** - Use while loops to skip all duplicates
- **Choose correct comparison** - `<` for non-overlapping, `<=` for overlapping allowed
- **Test with examples** - Walk through small examples step by step
- **Consider edge cases** - Empty, single element, all same, negative numbers

## Complexity Analysis

### Time Complexity:
- **Opposite Directional:** O(n) - Each pointer moves at most n times
- **Same Directional:** O(n) - Fast pointer traverses once
- **With Sorting:** O(n log n) - Dominated by sort operation
- **3Sum/4Sum:** O(n²)/O(n³) - One/two fixed elements + two pointers

### Space Complexity:
- Usually **O(1)** - Only using pointer variables
- May need **O(n)** for output in some problems
- **O(log n)** to **O(n)** if sorting is needed (depending on sort algorithm)

## When to Use Two Pointers

**Use Two Pointers when:**
- ✅ Array/list is sorted or can be sorted
- ✅ Need to find pairs/triplets with certain properties
- ✅ Need in-place array modifications
- ✅ Need to partition or rearrange elements
- ✅ Working with linked lists (fast & slow)
- ✅ Need to compare elements from different positions

**Don't Use Two Pointers when:**
- ❌ Need to maintain original order and can't sort
- ❌ Elements don't need to be contiguous and can be anywhere
- ❌ Need to check all possible combinations (might need other approaches)
- ❌ Problem requires checking each element with all others

## Problem Categories for Practice

### 1. Array Two Pointers
- Two Sum II (sorted array)
- 3Sum, 4Sum
- Remove duplicates from sorted array
- Move zeros
- Sort colors (Dutch National Flag)
- Squares of sorted array

### 2. String Two Pointers
- Valid palindrome
- Reverse string
- Reverse words in a string
- String compression
- Backspace string compare

### 3. Linked List Two Pointers
- Linked list cycle detection
- Find middle of linked list
- Remove nth node from end
- Palindrome linked list
- Intersection of two linked lists

### 4. Array Manipulation
- Container with most water
- Trapping rain water
- Partition array
- Sort colors

### 5. Subarray Problems (Sliding Window Variant)
- Longest substring without repeating characters
- Minimum size subarray sum
- Fruits into baskets

## Comparison with Other Techniques

| Technique | When to Use | Time | Space |
|-----------|-------------|------|-------|
| **Two Pointers** | Sorted data, pairs, in-place | O(n) | O(1) |
| **Hash Map** | Unsorted data, any pairs | O(n) | O(n) |
| **Sliding Window** | Contiguous subarrays | O(n) | O(1)-O(k) |
| **Binary Search** | Sorted data, single element | O(log n) | O(1) |

## Summary

Two Pointers is a powerful technique that reduces time complexity by intelligently moving pointers instead of using nested loops. Master these four patterns:

1. **Opposite Directional** - Converge from both ends
2. **Same Directional** - Fast & slow pointers
3. **Three Pointers** - Extension for triplet problems
4. **Partitioning** - Rearranging elements

**Key Takeaway:** Ask yourself - "Can I eliminate options by comparing elements at different positions?" If yes, two pointers might be your solution!

## Quick Reference Guide

```
Opposite Direction:     left → ← right
Same Direction:         slow → fast →
Three Pointers:         i → left → ← right
Partitioning:          low → mid → ← high
```

Practice these patterns, understand when to apply each, and you'll solve two-pointer problems efficiently and elegantly!
