# HashMap (Hash Table): Complete Guide

## What is a HashMap?

A HashMap (also called Hash Table or Dictionary) is a data structure that stores key-value pairs and provides fast lookup, insertion, and deletion operations. It uses a hash function to compute an index into an array of buckets or slots.

Think of it like a phone book where you can instantly find a person's number by their name, without searching through the entire book.

## Core Principles

**How HashMap Works:**
1. **Hash Function** - Converts key into array index
2. **Bucket Array** - Stores key-value pairs
3. **Collision Handling** - Resolves when two keys hash to same index
4. **Dynamic Resizing** - Expands when load factor exceeds threshold

**Basic Operations:**
- **Insert/Put** - Add or update key-value pair - Average O(1)
- **Get/Lookup** - Retrieve value by key - Average O(1)
- **Delete/Remove** - Remove key-value pair - Average O(1)
- **Contains** - Check if key exists - Average O(1)

## Visual Representation

```
Hash Function: hash(key) % array_size

Keys: "apple", "banana", "cherry"

Array Index:  0    1    2    3    4    5
             [  ] [  ] [  ] [  ] [  ] [  ]
                  ↑         ↑         ↑
              "banana"  "apple"  "cherry"
              (price)   (price)   (price)

Collision: If hash("grape") also maps to index 1
→ Use chaining (linked list) or open addressing
```

## HashMap Implementation (From Scratch)

### Implementation: Separate Chaining

```python
class HashMap:
    """
    HashMap using separate chaining for collision resolution
    """
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        """Hash function to compute bucket index"""
        return hash(key) % self.capacity
    
    def put(self, key, value):
        """
        Insert or update key-value pair
        Time: O(1) average, O(n) worst case
        """
        index = self._hash(key)
        bucket = self.buckets[index]
        
        # Update if key exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Insert new key-value pair
        bucket.append((key, value))
        self.size += 1
        
        # Resize if needed
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize()
    
    def get(self, key):
        """
        Retrieve value by key
        Time: O(1) average, O(n) worst case
        """
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def remove(self, key):
        """
        Remove key-value pair
        Time: O(1) average, O(n) worst case
        """
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """Check if key exists"""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def _resize(self):
        """Double the capacity and rehash all elements"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def keys(self):
        """Return all keys"""
        result = []
        for bucket in self.buckets:
            for key, _ in bucket:
                result.append(key)
        return result
    
    def values(self):
        """Return all values"""
        result = []
        for bucket in self.buckets:
            for _, value in bucket:
                result.append(value)
        return result
    
    def items(self):
        """Return all key-value pairs"""
        result = []
        for bucket in self.buckets:
            for item in bucket:
                result.append(item)
        return result

# Example usage
hashmap = HashMap()
hashmap.put("apple", 5)
hashmap.put("banana", 3)
hashmap.put("cherry", 7)
print(hashmap.get("banana"))  # 3
print(hashmap.contains("apple"))  # True
hashmap.remove("apple")
print(hashmap.contains("apple"))  # False
```

## Using Python's Built-in Dictionary

Python's `dict` is a highly optimized hash table implementation.

```python
# Creating dictionaries
hashmap = {}
hashmap = dict()
hashmap = {"key1": "value1", "key2": "value2"}

# Basic operations
hashmap["key"] = "value"  # Insert/Update - O(1)
value = hashmap["key"]    # Get - O(1), raises KeyError if missing
value = hashmap.get("key", default_value)  # Get with default - O(1)
del hashmap["key"]        # Delete - O(1)
exists = "key" in hashmap # Check existence - O(1)

# Iteration
for key in hashmap:
    print(key, hashmap[key])

for key, value in hashmap.items():
    print(key, value)

# Methods
hashmap.keys()    # Get all keys
hashmap.values()  # Get all values
hashmap.items()   # Get all key-value pairs
hashmap.pop(key)  # Remove and return value
hashmap.clear()   # Remove all items
```

## Using collections.defaultdict

Automatically creates default values for missing keys.

```python
from collections import defaultdict

# Default value is 0
freq = defaultdict(int)
freq['apple'] += 1  # No KeyError, starts at 0

# Default value is empty list
groups = defaultdict(list)
groups['fruits'].append('apple')

# Default value is empty set
unique = defaultdict(set)
unique['colors'].add('red')

# Custom default
hashmap = defaultdict(lambda: "default_value")
```

## Using collections.Counter

Specialized dictionary for counting.

```python
from collections import Counter

# Count elements
nums = [1, 2, 2, 3, 3, 3]
counter = Counter(nums)
print(counter)  # Counter({3: 3, 2: 2, 1: 1})

# Most common elements
print(counter.most_common(2))  # [(3, 3), (2, 2)]

# String counting
text = "hello world"
char_count = Counter(text)
print(char_count['l'])  # 3

# Operations
c1 = Counter(['a', 'b', 'c'])
c2 = Counter(['a', 'b', 'd'])
print(c1 + c2)  # Counter({'a': 2, 'b': 2, 'c': 1, 'd': 1})
print(c1 - c2)  # Counter({'c': 1})
```

## Pattern 1: Frequency Counting

### Example 1: Two Sum

**Problem:** Find two indices where numbers sum to target.

```python
def twoSum(nums, target):
    """
    Time: O(n), Space: O(n)
    """
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []

# Example usage
print(twoSum([2, 7, 11, 15], 9))  # [0, 1]
print(twoSum([3, 2, 4], 6))       # [1, 2]
```

### Example 2: Group Anagrams

**Problem:** Group strings that are anagrams of each other.

```python
def groupAnagrams(strs):
    """
    Time: O(n × k log k), Space: O(n × k)
    where n is number of strings, k is max length
    """
    from collections import defaultdict
    
    anagrams = defaultdict(list)
    
    for s in strs:
        # Sort string as key
        key = ''.join(sorted(s))
        anagrams[key].append(s)
    
    return list(anagrams.values())

# Alternative: Character count as key
def groupAnagrams_count(strs):
    """
    Time: O(n × k), Space: O(n × k)
    """
    from collections import defaultdict
    
    anagrams = defaultdict(list)
    
    for s in strs:
        # Use character count as key
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        anagrams[tuple(count)].append(s)
    
    return list(anagrams.values())

# Example usage
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(groupAnagrams(strs))
# [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

### Example 3: Top K Frequent Elements

**Problem:** Find k most frequent elements.

```python
from collections import Counter
import heapq

def topKFrequent(nums, k):
    """
    Time: O(n log k), Space: O(n)
    """
    freq = Counter(nums)
    return heapq.nlargest(k, freq.keys(), key=freq.get)

# Alternative: Bucket sort O(n)
def topKFrequent_bucket(nums, k):
    """
    Time: O(n), Space: O(n)
    """
    freq = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    
    # Place numbers in buckets by frequency
    for num, count in freq.items():
        buckets[count].append(num)
    
    # Collect k most frequent
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    
    return result

# Example usage
print(topKFrequent([1, 1, 1, 2, 2, 3], 2))  # [1, 2]
```

## Pattern 2: Tracking Indices/Positions

### Example 1: First Unique Character

**Problem:** Find index of first non-repeating character.

```python
def firstUniqChar(s):
    """
    Time: O(n), Space: O(1) - only 26 letters
    """
    from collections import Counter
    
    freq = Counter(s)
    
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    
    return -1

# Example usage
print(firstUniqChar("leetcode"))     # 0
print(firstUniqChar("loveleetcode")) # 2
```

### Example 2: Subarray Sum Equals K

**Problem:** Count subarrays with sum equal to k.

```python
def subarraySum(nums, k):
    """
    Time: O(n), Space: O(n)
    Uses prefix sum + hashmap
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # prefix_sum -> frequency
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if (prefix_sum - k) in sum_freq:
            count += sum_freq[prefix_sum - k]
        
        # Update frequency
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count

# Example usage
print(subarraySum([1, 1, 1], 2))      # 2
print(subarraySum([1, 2, 3], 3))      # 2
```

### Example 3: Longest Consecutive Sequence

**Problem:** Find length of longest consecutive sequence in unsorted array.

```python
def longestConsecutive(nums):
    """
    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Only start counting if it's the beginning of sequence
        if num - 1 not in num_set:
            current = num
            length = 1
            
            # Count consecutive numbers
            while current + 1 in num_set:
                current += 1
                length += 1
            
            longest = max(longest, length)
    
    return longest

# Example usage
print(longestConsecutive([100, 4, 200, 1, 3, 2]))  # 4 (1,2,3,4)
print(longestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))  # 9
```

## Pattern 3: Sliding Window with HashMap

### Example 1: Longest Substring Without Repeating Characters

**Problem:** Find length of longest substring with unique characters.

```python
def lengthOfLongestSubstring(s):
    """
    Time: O(n), Space: O(min(n, charset_size))
    """
    char_index = {}  # char -> last seen index
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        # If char seen and in current window
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Example usage
print(lengthOfLongestSubstring("abcabcbb"))  # 3 ("abc")
print(lengthOfLongestSubstring("bbbbb"))     # 1 ("b")
print(lengthOfLongestSubstring("pwwkew"))    # 3 ("wke")
```

### Example 2: Minimum Window Substring

**Problem:** Find minimum window containing all characters of target.

```python
def minWindow(s, t):
    """
    Time: O(m + n), Space: O(k) where k is unique chars in t
    """
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
    
    for right, char in enumerate(s):
        # Add character to window
        window_counts[char] = window_counts.get(char, 0) + 1
        
        # Check if frequency matches
        if char in target_count and window_counts[char] == target_count[char]:
            formed += 1
        
        # Try to shrink window
        while left <= right and formed == required:
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_window = (left, right)
            
            # Remove leftmost character
            char = s[left]
            window_counts[char] -= 1
            if char in target_count and window_counts[char] < target_count[char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_window[0]:min_window[1] + 1]

# Example usage
print(minWindow("ADOBECODEBANC", "ABC"))  # "BANC"
```

### Example 3: Find All Anagrams

**Problem:** Find all start indices of anagrams of p in s.

```python
def findAnagrams(s, p):
    """
    Time: O(n), Space: O(1) - only 26 letters
    """
    from collections import Counter
    
    if len(p) > len(s):
        return []
    
    p_count = Counter(p)
    s_count = Counter(s[:len(p)])
    result = []
    
    if s_count == p_count:
        result.append(0)
    
    for i in range(len(p), len(s)):
        # Add new character
        s_count[s[i]] += 1
        
        # Remove old character
        s_count[s[i - len(p)]] -= 1
        if s_count[s[i - len(p)]] == 0:
            del s_count[s[i - len(p)]]
        
        # Check if anagram
        if s_count == p_count:
            result.append(i - len(p) + 1)
    
    return result

# Example usage
print(findAnagrams("cbaebabacd", "abc"))  # [0, 6]
print(findAnagrams("abab", "ab"))         # [0, 1, 2]
```

## Pattern 4: Graph Problems with HashMap

### Example 1: Clone Graph

**Problem:** Deep copy an undirected graph.

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []

def cloneGraph(node):
    """
    Time: O(V + E), Space: O(V)
    """
    if not node:
        return None
    
    clones = {}  # original -> clone mapping
    
    def dfs(node):
        if node in clones:
            return clones[node]
        
        # Create clone
        clone = Node(node.val)
        clones[node] = clone
        
        # Clone neighbors
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

### Example 2: Course Schedule (Cycle Detection)

**Problem:** Check if all courses can be finished (detect cycle in directed graph).

```python
def canFinish(numCourses, prerequisites):
    """
    Time: O(V + E), Space: O(V + E)
    """
    from collections import defaultdict
    
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # 0: unvisited, 1: visiting, 2: visited
    state = [0] * numCourses
    
    def has_cycle(course):
        if state[course] == 1:  # Visiting - cycle detected
            return True
        if state[course] == 2:  # Already visited
            return False
        
        state[course] = 1  # Mark as visiting
        
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True
        
        state[course] = 2  # Mark as visited
        return False
    
    for course in range(numCourses):
        if has_cycle(course):
            return False
    
    return True

# Example usage
print(canFinish(2, [[1, 0]]))           # True
print(canFinish(2, [[1, 0], [0, 1]]))   # False
```

## Pattern 5: Design Problems

### Example 1: LRU Cache

**Problem:** Design Least Recently Used cache with O(1) operations.

```python
class LRUCache:
    """
    Time: O(1) for get and put
    Space: O(capacity)
    """
    class Node:
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Dummy head and tail for easier manipulation
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        """Remove node from linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node):
        """Add node right after head"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def _move_to_head(self, node):
        """Move existing node to head"""
        self._remove(node)
        self._add_to_head(node)
    
    def get(self, key):
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._move_to_head(node)
        return node.value
    
    def put(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            node = self.Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            
            if len(self.cache) > self.capacity:
                # Remove least recently used (tail.prev)
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]

# Example usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))    # 1
cache.put(3, 3)        # Evicts key 2
print(cache.get(2))    # -1 (not found)
```

### Example 2: Design HashMap

**Problem:** (Already shown in implementation section above)

### Example 3: Randomized Set

**Problem:** Design structure with insert, delete, getRandom in O(1).

```python
import random

class RandomizedSet:
    """
    All operations: O(1) average
    """
    def __init__(self):
        self.data = []  # Store values
        self.index_map = {}  # value -> index
    
    def insert(self, val):
        if val in self.index_map:
            return False
        
        self.data.append(val)
        self.index_map[val] = len(self.data) - 1
        return True
    
    def remove(self, val):
        if val not in self.index_map:
            return False
        
        # Swap with last element
        idx = self.index_map[val]
        last_val = self.data[-1]
        
        self.data[idx] = last_val
        self.index_map[last_val] = idx
        
        # Remove last element
        self.data.pop()
        del self.index_map[val]
        
        return True
    
    def getRandom(self):
        return random.choice(self.data)

# Example usage
rs = RandomizedSet()
rs.insert(1)
rs.insert(2)
rs.insert(3)
print(rs.getRandom())  # Random: 1, 2, or 3
rs.remove(2)
print(rs.getRandom())  # Random: 1 or 3
```

## Pattern 6: String Pattern Matching

### Example 1: Isomorphic Strings

**Problem:** Check if two strings are isomorphic (character mapping exists).

```python
def isIsomorphic(s, t):
    """
    Time: O(n), Space: O(1) - at most 256 chars
    """
    if len(s) != len(t):
        return False
    
    s_to_t = {}
    t_to_s = {}
    
    for char_s, char_t in zip(s, t):
        # Check s -> t mapping
        if char_s in s_to_t:
            if s_to_t[char_s] != char_t:
                return False
        else:
            s_to_t[char_s] = char_t
        
        # Check t -> s mapping
        if char_t in t_to_s:
            if t_to_s[char_t] != char_s:
                return False
        else:
            t_to_s[char_t] = char_s
    
    return True

# Example usage
print(isIsomorphic("egg", "add"))    # True
print(isIsomorphic("foo", "bar"))    # False
print(isIsomorphic("paper", "title")) # True
```

### Example 2: Word Pattern

**Problem:** Check if pattern matches string (full word matching).

```python
def wordPattern(pattern, s):
    """
    Time: O(n), Space: O(n)
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    char_to_word = {}
    word_to_char = {}
    
    for char, word in zip(pattern, words):
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            char_to_word[char] = word
        
        if word in word_to_char:
            if word_to_char[word] != char:
                return False
        else:
            word_to_char[word] = char
    
    return True

# Example usage
print(wordPattern("abba", "dog cat cat dog"))    # True
print(wordPattern("abba", "dog cat cat fish"))   # False
print(wordPattern("aaaa", "dog cat cat dog"))    # False
```

## Common Pitfalls & Tips

### Pitfalls to Avoid:
- **KeyError** - Always check if key exists or use `.get()`
- **Modifying during iteration** - Can cause runtime error
- **Using mutable keys** - Lists/dicts can't be keys (use tuples)
- **Not handling collisions** - In custom implementations
- **Forgetting default values** - Use `defaultdict` or `.get()`
- **Hash function quality** - Poor hash leads to many collisions

### Pro Tips:
- **Use `.get()` with default** - `dict.get(key, default_value)`
- **defaultdict for counters** - `defaultdict(int)` for frequency
- **Counter for counting** - Specialized for frequency problems
- **Set for O(1) lookup** - When only checking existence
- **Tuple keys for 2D** - `hashmap[(row, col)]` for grids
- **Use `in` operator** - More Pythonic than `.has_key()`
- **Dictionary comprehension** - `{k: v for k, v in items}`

## Complexity Analysis

### Time Complexity:
| Operation | Average | Worst Case |
|-----------|---------|------------|
| Insert | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Search | O(1) | O(n) |
| Space | O(n) | O(n) |

**Worst case** occurs with many collisions (poor hash function)

### Load Factor:
- **Load Factor** = n / m (items / buckets)
- **Typical threshold** = 0.75
- **When exceeded** = Resize (double capacity and rehash)

## When to Use HashMap

**Use HashMap when:**
- ✅ Need fast lookup by key
- ✅ Counting frequencies
- ✅ Tracking seen elements
- ✅ Caching/memoization
- ✅ Grouping/categorizing data
- ✅ Finding duplicates
- ✅ Two sum / pair finding problems
- ✅ Mapping relationships

**Don't Use HashMap when:**
- ❌ Need ordered iteration (use OrderedDict or TreeMap)
- ❌ Need range queries (use TreeMap/BST)
- ❌ Memory is very constrained
- ❌ Hash collisions are common (poor hash function)
- ❌ Need to maintain sorted order

## HashMap vs Other Data Structures

| Structure | Search | Insert | Delete | Ordered |
|-----------|--------|--------|--------|---------|
| **HashMap** | O(1)* | O(1)* | O(1)* | No |
| **Array** | O(n) | O(1)** | O(n) | Yes (by index) |
| **LinkedList** | O(n) | O(1)*** | O(1)*** | Yes (insertion) |
| **BST** | O(log n) | O(log n) | O(log n) | Yes |
| **Set** | O(1)* | O(1)* | O(1)* | No |

*Average case, **at end, ***with pointer

## Problem Categories for Practice

### 1. Frequency Counting
- Two sum
- Top K frequent elements
- Group anagrams
- Valid anagram
- Find duplicates

### 2. Substring/Subarray Problems
- Longest substring without repeating
- Minimum window substring
- Subarray sum equals K
- Find all anagrams
- Longest consecutive sequence

### 3. Design Problems
- LRU cache
- Design HashMap
- Randomized set
- Time-based key-value store
- Insert delete getRandom O(1)

### 4. Graph/Tree Problems
- Clone graph
- Course schedule
- Word ladder
- Accounts merge
- Evaluate division

### 5. String Matching
- Isomorphic strings
- Word pattern
- Valid Sudoku
- Happy number
- Contains duplicate

### 6. Array Problems
- Intersection of arrays
- Single number
- Missing number
- First missing positive
- Majority element

## Quick Implementation Reference

### Basic Dictionary Operations
```python
# Create
hashmap = {}
hashmap = dict()

# Insert/Update
hashmap[key] = value

# Get
value = hashmap[key]  # Raises KeyError if missing
value = hashmap.get(key, default)

# Delete
del hashmap[key]
value = hashmap.pop(key, default)

# Check existence
if key in hashmap:
    pass

# Iteration
for key in hashmap:
    print(key, hashmap[key])

for key, value in hashmap.items():
    print(key, value)
```

### Common Patterns
```python
# Frequency counting
from collections import Counter
freq = Counter(arr)

# Default values
from collections import defaultdict
counts = defaultdict(int)
groups = defaultdict(list)

# Two pointers with hashmap
seen = {}
for i, num in enumerate(nums):
    if target - num in seen:
        return [seen[target - num], i]
    seen[num] = i

# Sliding window with hashmap
char_count = {}
for char in s:
    char_count[char] = char_count.get(char, 0) + 1
```

## Advanced Techniques

### 1. Tuple as Key (for 2D problems)
```python
# Grid problems
visited = set()
visited.add((row, col))

# Multiple values as key
cache = {}
cache[(x, y, z)] = result
```

### 2. Sorting for Anagrams
```python
# Use sorted string as key
anagrams = {}
for word in words:
    key = ''.join(sorted(word))
    anagrams.setdefault(key, []).append(word)
```

### 3. Rolling Hash (Rabin-Karp)
```python
def rolling_hash(s, pattern_len):
    """
    Efficient substring search
    """
    base = 26
    mod = 10**9 + 7
    
    hash_val = 0
    for char in s[:pattern_len]:
        hash_val = (hash_val * base + ord(char)) % mod
    return hash_val
```

### 4. Bidirectional Mapping
```python
# When need to map both ways
char_to_word = {}
word_to_char = {}

# Or use single dict with tuples
mapping = {}
mapping[('char', c)] = word
mapping[('word', word)] = c
```

## Summary

HashMap is one of the most fundamental and versatile data structures:

**Key Characteristics:**
- **O(1) operations** - Insert, delete, search (average case)
- **Hash function** - Maps keys to array indices
- **Collision handling** - Chaining or open addressing
- **Dynamic resizing** - Maintains performance as size grows

**Common Applications:**
1. **Frequency counting** - Character/element frequencies
2. **Two sum problems** - Finding pairs with target sum
3. **Caching** - Store computed results (memoization)
4. **Grouping** - Anagrams, similar items
5. **Tracking seen** - Detect duplicates, visited nodes
6. **Mapping** - Isomorphic strings, word patterns
7. **Graph problems** - Adjacency lists, node mappings

**Key Patterns:**
1. **Frequency map** - Count occurrences
2. **Index map** - Track positions
3. **Sliding window** - With character/element tracking
4. **Two pointers** - With complement lookup
5. **Prefix sum** - With sum frequency tracking

**Python HashMap Tools:**
- `dict` - Standard dictionary
- `defaultdict` - Auto-creates default values
- `Counter` - Specialized for counting
- `OrderedDict` - Maintains insertion order (Python 3.7+ dicts are ordered)

**Key Takeaway:** Master HashMap for fast lookups and frequency counting. It's essential for optimizing algorithms from O(n²) to O(n). Understanding when and how to use HashMaps is crucial for solving many interview problems efficiently!

Practice these patterns and you'll handle hash-based problems with confidence and optimal time complexity!
