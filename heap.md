# Priority Queue & Heap: Complete Guide

## What is a Priority Queue?

A Priority Queue is an abstract data type where each element has a priority associated with it. Elements are dequeued based on their priority rather than their insertion order. The element with the highest (or lowest) priority is always removed first.

Think of it like an emergency room - patients are treated based on severity (priority), not arrival time.

## What is a Heap?

A Heap is a complete binary tree that satisfies the heap property. It's the most common implementation of a Priority Queue.

**Heap Property:**
- **Min Heap:** Parent ≤ Children (smallest element at root)
- **Max Heap:** Parent ≥ Children (largest element at root)

**Complete Binary Tree:** All levels filled except possibly the last, which is filled from left to right.

## Visual Representation

### Min Heap Example
```
        1
       / \
      3   2
     / \ / \
    7  5 4  6
    
Array: [1, 3, 2, 7, 5, 4, 6]
Index:  0  1  2  3  4  5  6
```

### Max Heap Example
```
        10
       /  \
      8    9
     / \  / \
    4  5 6   7
    
Array: [10, 8, 9, 4, 5, 6, 7]
Index:   0  1  2  3  4  5  6
```

## Heap Relationships (Array-based)

For element at index `i`:
- **Parent:** `(i - 1) // 2`
- **Left Child:** `2 * i + 1`
- **Right Child:** `2 * i + 2`

## Core Operations

1. **Insert (Push)** - Add element - O(log n)
2. **Extract-Min/Max (Pop)** - Remove root - O(log n)
3. **Peek (Top)** - View root - O(1)
4. **Heapify** - Convert array to heap - O(n)

## Implementation: Min Heap (From Scratch)

```python
class MinHeap:
    """
    Min Heap implementation using array
    Parent is smaller than both children
    """
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, key):
        """
        Insert element and maintain heap property
        Time: O(log n)
        """
        # Add to end of heap
        self.heap.append(key)
        
        # Bubble up to maintain heap property
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """
        Move element up until heap property is satisfied
        """
        parent = self.parent(i)
        
        # If current element is smaller than parent, swap
        if i > 0 and self.heap[i] < self.heap[parent]:
            self.swap(i, parent)
            self._heapify_up(parent)
    
    def extract_min(self):
        """
        Remove and return minimum element (root)
        Time: O(log n)
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        # Store root to return
        min_val = self.heap[0]
        
        # Move last element to root
        self.heap[0] = self.heap.pop()
        
        # Bubble down to maintain heap property
        self._heapify_down(0)
        
        return min_val
    
    def _heapify_down(self, i):
        """
        Move element down until heap property is satisfied
        """
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        # Find smallest among parent, left child, right child
        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right
        
        # If smallest is not parent, swap and continue
        if min_index != i:
            self.swap(i, min_index)
            self._heapify_down(min_index)
    
    def peek(self):
        """
        Return minimum element without removing
        Time: O(1)
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def build_heap(self, arr):
        """
        Build heap from array
        Time: O(n)
        """
        self.heap = arr.copy()
        # Start from last non-leaf node and heapify down
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)

# Example usage
min_heap = MinHeap()
min_heap.insert(5)
min_heap.insert(3)
min_heap.insert(7)
min_heap.insert(1)
print(min_heap.peek())        # 1
print(min_heap.extract_min()) # 1
print(min_heap.extract_min()) # 3
```

## Implementation: Max Heap (From Scratch)

```python
class MaxHeap:
    """
    Max Heap implementation using array
    Parent is larger than both children
    """
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, key):
        """
        Insert element and maintain heap property
        Time: O(log n)
        """
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """
        Move element up until heap property is satisfied
        """
        parent = self.parent(i)
        
        # If current element is larger than parent, swap
        if i > 0 and self.heap[i] > self.heap[parent]:
            self.swap(i, parent)
            self._heapify_up(parent)
    
    def extract_max(self):
        """
        Remove and return maximum element (root)
        Time: O(log n)
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return max_val
    
    def _heapify_down(self, i):
        """
        Move element down until heap property is satisfied
        """
        max_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        # Find largest among parent, left child, right child
        if left < len(self.heap) and self.heap[left] > self.heap[max_index]:
            max_index = left
        
        if right < len(self.heap) and self.heap[right] > self.heap[max_index]:
            max_index = right
        
        # If largest is not parent, swap and continue
        if max_index != i:
            self.swap(i, max_index)
            self._heapify_down(max_index)
    
    def peek(self):
        """
        Return maximum element without removing
        Time: O(1)
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def build_heap(self, arr):
        """
        Build heap from array
        Time: O(n)
        """
        self.heap = arr.copy()
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)

# Example usage
max_heap = MaxHeap()
max_heap.insert(5)
max_heap.insert(3)
max_heap.insert(7)
max_heap.insert(1)
print(max_heap.peek())        # 7
print(max_heap.extract_max()) # 7
print(max_heap.extract_max()) # 5
```

## Using Python's heapq (Built-in)

Python's `heapq` module implements a min heap by default.

### Min Heap with heapq

```python
import heapq

# Create empty heap
min_heap = []

# Insert elements - O(log n)
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 7)
heapq.heappush(min_heap, 1)

# Peek at minimum - O(1)
print(min_heap[0])  # 1

# Extract minimum - O(log n)
min_val = heapq.heappop(min_heap)  # 1

# Build heap from list - O(n)
arr = [5, 3, 7, 1]
heapq.heapify(arr)
print(arr)  # [1, 3, 7, 5]

# Get n smallest elements
nums = [5, 3, 7, 1, 9, 2]
smallest_3 = heapq.nsmallest(3, nums)  # [1, 2, 3]

# Get n largest elements
largest_3 = heapq.nlargest(3, nums)  # [9, 7, 5]
```

### Max Heap with heapq (Using Negation)

```python
import heapq

# Create max heap by negating values
max_heap = []

# Insert - negate to simulate max heap
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)
heapq.heappush(max_heap, -1)

# Peek at maximum
print(-max_heap[0])  # 7

# Extract maximum
max_val = -heapq.heappop(max_heap)  # 7

# Alternative: Use tuple for complex objects
# heapq sorts by first element of tuple
max_heap = []
heapq.heappush(max_heap, (-priority, value))
```

## Heap Sort Algorithm

```python
def heap_sort(arr):
    """
    Sort array using heap
    Time: O(n log n), Space: O(1) in-place
    """
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify_down(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        # Move current root to end
        arr[0], arr[i] = arr[i], arr[0]
        # Heapify reduced heap
        heapify_down(arr, i, 0)
    
    return arr

def heapify_down(arr, n, i):
    """Helper function for heap sort"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_down(arr, n, largest)

# Example usage
arr = [12, 11, 13, 5, 6, 7]
sorted_arr = heap_sort(arr)
print(sorted_arr)  # [5, 6, 7, 11, 12, 13]
```

## Pattern 1: Top K Elements

### Example 1: Kth Largest Element

**Problem:** Find the kth largest element in an array.

**Approach 1: Using Custom Min Heap**
```python
def findKthLargest_custom(nums, k):
    """
    Time: O(n log k), Space: O(k)
    """
    min_heap = MinHeap()
    
    for num in nums:
        min_heap.insert(num)
        if min_heap.size() > k:
            min_heap.extract_min()
    
    return min_heap.peek()
```

**Approach 2: Using heapq**
```python
import heapq

def findKthLargest(nums, k):
    """
    Time: O(n log k), Space: O(k)
    """
    min_heap = []
    
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return min_heap[0]

# Alternative: O(n + k log n)
def findKthLargest_alt(nums, k):
    return heapq.nlargest(k, nums)[-1]

# Example usage
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(findKthLargest(nums, k))  # 5
```

### Example 2: K Closest Points to Origin

**Problem:** Find k points closest to origin (0, 0).

**Approach 1: Using Custom Max Heap**
```python
def kClosest_custom(points, k):
    """
    Time: O(n log k), Space: O(k)
    """
    # Max heap of (distance, point)
    max_heap = MaxHeap()
    
    for x, y in points:
        dist = x*x + y*y
        max_heap.insert((dist, [x, y]))
        if max_heap.size() > k:
            max_heap.extract_max()
    
    return [point for _, point in max_heap.heap]
```

**Approach 2: Using heapq**
```python
import heapq

def kClosest(points, k):
    """
    Time: O(n log k), Space: O(k)
    """
    max_heap = []
    
    for x, y in points:
        dist = -(x*x + y*y)  # Negate for max heap
        heapq.heappush(max_heap, (dist, [x, y]))
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    
    return [point for _, point in max_heap]

# Example usage
points = [[1, 3], [-2, 2], [5, 8], [0, 1]]
k = 2
print(kClosest(points, k))  # [[-2, 2], [0, 1]] or [[0, 1], [-2, 2]]
```

### Example 3: Top K Frequent Elements

**Problem:** Find k most frequent elements.

**Approach 1: Using Custom Max Heap**
```python
from collections import Counter

def topKFrequent_custom(nums, k):
    """
    Time: O(n log k), Space: O(n)
    """
    freq = Counter(nums)
    max_heap = MaxHeap()
    
    for num, count in freq.items():
        max_heap.insert((count, num))
    
    result = []
    for _ in range(k):
        count, num = max_heap.extract_max()
        result.append(num)
    
    return result
```

**Approach 2: Using heapq**
```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    """
    Time: O(n log k), Space: O(n)
    """
    freq = Counter(nums)
    # Use nlargest for cleaner code
    return heapq.nlargest(k, freq.keys(), key=freq.get)

# Alternative with manual heap
def topKFrequent_manual(nums, k):
    freq = Counter(nums)
    # Negate frequency for max heap behavior
    heap = [(-count, num) for num, count in freq.items()]
    heapq.heapify(heap)
    
    return [heapq.heappop(heap)[1] for _ in range(k)]

# Example usage
nums = [1, 1, 1, 2, 2, 3]
k = 2
print(topKFrequent(nums, k))  # [1, 2]
```

## Pattern 2: Merge K Sorted Lists/Arrays

### Example 1: Merge K Sorted Lists

**Problem:** Merge k sorted linked lists.

**Approach 1: Using Custom Min Heap**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists_custom(lists):
    """
    Time: O(n log k), Space: O(k)
    where n is total nodes, k is number of lists
    """
    min_heap = MinHeap()
    
    # Add first node from each list
    for i, lst in enumerate(lists):
        if lst:
            min_heap.insert((lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while not min_heap.is_empty():
        val, i, node = min_heap.extract_min()
        current.next = node
        current = current.next
        
        if node.next:
            min_heap.insert((node.next.val, i, node.next))
    
    return dummy.next
```

**Approach 2: Using heapq**
```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists):
    """
    Time: O(n log k), Space: O(k)
    """
    min_heap = []
    
    # Add first node from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))
    
    return dummy.next
```

### Example 2: Kth Smallest Element in Sorted Matrix

**Problem:** Find kth smallest in n×n matrix where each row and column is sorted.

```python
import heapq

def kthSmallest(matrix, k):
    """
    Time: O(k log n), Space: O(n)
    """
    n = len(matrix)
    min_heap = []
    
    # Add first element from each row
    for r in range(min(k, n)):
        heapq.heappush(min_heap, (matrix[r][0], r, 0))
    
    result = 0
    for _ in range(k):
        result, r, c = heapq.heappop(min_heap)
        
        # Add next element from same row
        if c + 1 < n:
            heapq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))
    
    return result

# Example usage
matrix = [
    [1,  5,  9],
    [10, 11, 13],
    [12, 13, 15]
]
k = 8
print(kthSmallest(matrix, k))  # 13
```

## Pattern 3: Two Heaps (Median Finding)

### Example: Find Median from Data Stream

**Problem:** Design a data structure that supports adding numbers and finding median.

**Approach 1: Using Custom Heaps**
```python
class MedianFinder_Custom:
    """
    Time: O(log n) for add, O(1) for median
    Space: O(n)
    """
    def __init__(self):
        self.small = MaxHeap()  # Lower half
        self.large = MinHeap()  # Upper half
    
    def addNum(self, num):
        # Add to appropriate heap
        if self.small.is_empty() or num <= self.small.peek():
            self.small.insert(num)
        else:
            self.large.insert(num)
        
        # Balance heaps (size difference ≤ 1)
        if self.small.size() > self.large.size() + 1:
            self.large.insert(self.small.extract_max())
        elif self.large.size() > self.small.size():
            self.small.insert(self.large.extract_min())
    
    def findMedian(self):
        if self.small.size() == self.large.size():
            return (self.small.peek() + self.large.peek()) / 2
        return self.small.peek()
```

**Approach 2: Using heapq**
```python
import heapq

class MedianFinder:
    """
    Time: O(log n) for add, O(1) for median
    Space: O(n)
    """
    def __init__(self):
        self.small = []  # Max heap (negate values)
        self.large = []  # Min heap
    
    def addNum(self, num):
        # Add to max heap (small)
        heapq.heappush(self.small, -num)
        
        # Balance: move largest from small to large
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        # Ensure small has equal or one more element
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2

# Example usage
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # 1.5
mf.addNum(3)
print(mf.findMedian())  # 2
```

## Pattern 4: Scheduling Problems

### Example: Task Scheduler

**Problem:** Schedule tasks with cooldown period n. Return minimum time needed.

```python
from collections import Counter
import heapq

def leastInterval(tasks, n):
    """
    Time: O(t log t), Space: O(26)
    """
    # Count frequencies
    freq = Counter(tasks)
    max_heap = [-count for count in freq.values()]
    heapq.heapify(max_heap)
    
    time = 0
    
    while max_heap:
        temp = []
        for _ in range(n + 1):
            if max_heap:
                count = heapq.heappop(max_heap)
                if count + 1 < 0:  # Still has tasks
                    temp.append(count + 1)
            time += 1
            
            if not max_heap and not temp:
                break
        
        # Push back to heap
        for count in temp:
            heapq.heappush(max_heap, count)
    
    return time

# Example usage
tasks = ["A", "A", "A", "B", "B", "B"]
n = 2
print(leastInterval(tasks, n))  # 8
```

## Pattern 5: Running Median/Statistics

### Example: Sliding Window Median

**Problem:** Find median in each sliding window of size k.

```python
import heapq
from collections import defaultdict

def medianSlidingWindow(nums, k):
    """
    Time: O(n log k), Space: O(k)
    """
    small = []  # Max heap
    large = []  # Min heap
    result = []
    remove_map = defaultdict(int)  # Lazy deletion
    
    def balance():
        # Move from small to large
        while len(small) > len(large) + 1:
            heapq.heappush(large, -heapq.heappop(small))
        # Move from large to small
        while len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))
    
    def get_median():
        if k % 2 == 1:
            return -small[0]
        return (-small[0] + large[0]) / 2
    
    def remove_top(heap):
        while heap and remove_map[abs(heap[0])] > 0:
            remove_map[abs(heapq.heappop(heap))] -= 1
    
    # Build initial window
    for i in range(k):
        heapq.heappush(small, -nums[i])
    
    for _ in range(k // 2):
        heapq.heappush(large, -heapq.heappop(small))
    
    result.append(get_median())
    
    # Slide window
    for i in range(k, len(nums)):
        out_num = nums[i - k]
        in_num = nums[i]
        
        # Track removal
        remove_map[out_num] += 1
        
        # Add new element
        if in_num <= -small[0]:
            heapq.heappush(small, -in_num)
        else:
            heapq.heappush(large, in_num)
        
        # Adjust balance based on removed element
        if out_num <= -small[0]:
            if len(small) > len(large) + 1:
                heapq.heappush(large, -heapq.heappop(small))
        else:
            if len(large) > len(small):
                heapq.heappush(small, -heapq.heappop(large))
        
        # Clean heaps
        remove_top(small)
        remove_top(large)
        balance()
        
        result.append(get_median())
    
    return result

# Example usage
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(medianSlidingWindow(nums, k))  # [1, -1, -1, 3, 5, 6]
```

## Common Pitfalls & Tips

### Pitfalls to Avoid:
- **Forgetting to negate for max heap** with heapq
- **Off-by-one errors** in parent/child index calculations
- **Not handling empty heap** before peek/pop
- **Wrong comparison** in heapify_up/down (< vs > for min/max)
- **Forgetting to heapify** after building heap from array
- **Not balancing** in two-heap problems

### Pro Tips:
- **Use heapq for simplicity** unless learning heap internals
- **Negate values** for max heap with heapq: `heapq.heappush(heap, -val)`
- **Use tuples** for priority + data: `(priority, data)`
- **Build heap is O(n)**, faster than n insertions O(n log n)
- **Two heaps pattern** for median/streaming statistics
- **Lazy deletion** with hashmap for sliding window problems
- **Keep heaps balanced** in two-heap problems (size difference ≤ 1)

## Complexity Analysis

### Time Complexity:
| Operation | Binary Heap | Fibonacci Heap |
|-----------|-------------|----------------|
| Insert | O(log n) | O(1) amortized |
| Extract-Min/Max | O(log n) | O(log n) amortized |
| Peek | O(1) | O(1) |
| Delete | O(log n) | O(log n) amortized |
| Build Heap | O(n) | O(n) |
| Decrease Key | O(log n) | O(1) amortized |

### Space Complexity:
- **All implementations:** O(n) where n is number of elements

## When to Use Priority Queue/Heap

**Use Priority Queue/Heap when:**
- ✅ Need to repeatedly get min/max element
- ✅ Finding top K elements
- ✅ Merging K sorted lists/arrays
- ✅ Scheduling problems with priorities
- ✅ Graph algorithms (Dijkstra, Prim's)
- ✅ Median finding in data stream
- ✅ Event simulation systems

**Don't Use Priority Queue when:**
- ❌ Need FIFO ordering (use Queue)
- ❌ Need random access to elements
- ❌ Need to search for arbitrary elements (O(n) in heap)
- ❌ Need to maintain sorted order of all elements (use BST)

## Heap vs Other Data Structures

| Structure | Find Min/Max | Insert | Delete Min/Max | Search |
|-----------|--------------|--------|----------------|--------|
| **Heap** | O(1) | O(log n) | O(log n) | O(n) |
| **BST** | O(log n) | O(log n) | O(log n) | O(log n) |
| **Sorted Array** | O(1) | O(n) | O(n) | O(log n) |
| **Unsorted Array** | O(n) | O(1) | O(n) | O(n) |

## Problem Categories for Practice

### 1. Top K Elements
- Kth largest element
- K closest points to origin
- Top K frequent elements
- K closest elements

### 2. Merge Problems
- Merge K sorted lists
- Kth smallest in sorted matrix
- Find K pairs with smallest sums
- Smallest range covering elements from K lists

### 3. Two Heaps (Median)
- Find median from data stream
- Sliding window median
- IPO (maximize capital)

### 4. Scheduling
- Task scheduler
- Meeting rooms II
- Single-threaded CPU
- Process tasks using servers

### 5. Graph Problems
- Dijkstra's shortest path
- Prim's minimum spanning tree
- Network delay time
- Path with minimum effort

### 6. Design Problems
- Design Twitter
- Exam room
- Seat reservation manager

## Quick Implementation Reference

### Custom Min Heap
