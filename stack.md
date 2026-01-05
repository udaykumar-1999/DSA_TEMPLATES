# Stack Data Structure: Complete Guide

## What is a Stack?

A Stack is a linear data structure that follows the **LIFO (Last-In-First-Out)** principle. The last element added to the stack will be the first one to be removed, just like a stack of plates.

Think of it like a stack of books - you can only add or remove books from the top, and the last book you placed is the first one you can take off.

## Core Principles

**LIFO (Last-In-First-Out):**
- Elements are inserted at the **top**
- Elements are removed from the **top**
- Only the top element is accessible

**Basic Operations:**
1. **Push** - Add element to top - O(1)
2. **Pop** - Remove element from top - O(1)
3. **Peek (Top)** - View top element without removing - O(1)
4. **isEmpty** - Check if stack is empty - O(1)
5. **Size** - Get number of elements - O(1)

## Visual Representation

```
Push(1), Push(2), Push(3):

    [3]  ← Top
    [2]
    [1]
    ---

Pop():

    [2]  ← Top (3 removed)
    [1]
    ---

Push(4):

    [4]  ← Top
    [2]
    [1]
    ---
```

## Stack Implementations

### Implementation 1: Using Python List

```python
class Stack:
    """
    Stack using Python list
    All operations: O(1)
    """
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """View top item without removing"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
    
    def clear(self):
        """Remove all elements"""
        self.items = []

# Example usage
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())   # 3
print(stack.peek())  # 2
print(stack.size())  # 2
```

### Implementation 2: Using Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedStack:
    """
    Stack using linked list
    All operations: O(1)
    Better for memory allocation
    """
    def __init__(self):
        self.top = None
        self.count = 0
    
    def push(self, item):
        """Add item to top"""
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self.count += 1
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        
        item = self.top.data
        self.top = self.top.next
        self.count -= 1
        return item
    
    def peek(self):
        """View top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.top.data
    
    def is_empty(self):
        return self.top is None
    
    def size(self):
        return self.count

# Example usage
stack = LinkedStack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())   # 3
print(stack.peek())  # 2
```

### Implementation 3: Using collections.deque

```python
from collections import deque

class DequeStack:
    """
    Stack using deque (most efficient)
    All operations: O(1)
    """
    def __init__(self):
        self.items = deque()
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Direct usage (most common)
stack = []
stack.append(1)     # Push
stack.append(2)
stack.append(3)
item = stack.pop()  # Pop
top = stack[-1]     # Peek
```

## Pattern 1: Balanced Parentheses

### Example 1: Valid Parentheses

**Problem:** Check if string has valid brackets: (), {}, []

**Solution Steps:**
1. Push opening brackets onto stack
2. For closing bracket, check if it matches top of stack
3. Stack should be empty at the end

```python
def isValid(s):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

# Example usage
print(isValid("()[]{}"))    # True
print(isValid("([)]"))      # False
print(isValid("{[]}"))      # True
```

### Example 2: Minimum Add to Make Valid Parentheses

**Problem:** Find minimum insertions to make string valid.

```python
def minAddToMakeValid(s):
    """
    Time: O(n), Space: O(1)
    """
    open_needed = 0   # Unmatched '('
    close_needed = 0  # Unmatched ')'
    
    for char in s:
        if char == '(':
            open_needed += 1
        elif char == ')':
            if open_needed > 0:
                open_needed -= 1
            else:
                close_needed += 1
    
    return open_needed + close_needed

# Example usage
print(minAddToMakeValid("())"))   # 1
print(minAddToMakeValid("((("))   # 3
```

### Example 3: Longest Valid Parentheses

**Problem:** Find length of longest valid parentheses substring.

```python
def longestValidParentheses(s):
    """
    Time: O(n), Space: O(n)
    """
    stack = [-1]  # Base for valid substring
    max_len = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                # No matching '(', use current as base
                stack.append(i)
            else:
                # Calculate length
                max_len = max(max_len, i - stack[-1])
    
    return max_len

# Example usage
print(longestValidParentheses("(()"))     # 2
print(longestValidParentheses(")()())"))  # 4
```

## Pattern 2: Expression Evaluation

### Example 1: Evaluate Reverse Polish Notation

**Problem:** Evaluate postfix expression (e.g., "2 1 + 3 *" = 9)

```python
def evalRPN(tokens):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                # Integer division towards zero
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]

# Example usage
print(evalRPN(["2", "1", "+", "3", "*"]))  # 9
print(evalRPN(["4", "13", "5", "/", "+"]))  # 6
```

### Example 2: Basic Calculator

**Problem:** Evaluate string expression with +, -, (, ).

```python
def calculate(s):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    num = 0
    sign = 1
    result = 0
    
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            # Push current result and sign onto stack
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            # Pop sign and previous result
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result
    
    result += sign * num
    return result

# Example usage
print(calculate("1 + 1"))        # 2
print(calculate(" 2-1 + 2 "))    # 3
print(calculate("(1+(4+5+2)-3)+(6+8)"))  # 23
```

### Example 3: Decode String

**Problem:** Decode string like "3[a2[c]]" → "accaccacc"

```python
def decodeString(s):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    current_num = 0
    current_str = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state onto stack
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            # Pop and decode
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char
    
    return current_str

# Example usage
print(decodeString("3[a]2[bc]"))     # "aaabcbc"
print(decodeString("3[a2[c]]"))      # "accaccacc"
print(decodeString("2[abc]3[cd]ef")) # "abcabccdcdcdef"
```

## Pattern 3: Monotonic Stack

A monotonic stack maintains elements in increasing or decreasing order.

### Example 1: Next Greater Element

**Problem:** For each element, find the next greater element to its right.

```python
def nextGreaterElement(nums):
    """
    Time: O(n), Space: O(n)
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop elements smaller than current
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result

# Example usage
print(nextGreaterElement([4, 5, 2, 10, 8]))  # [5, 10, 10, -1, -1]
print(nextGreaterElement([1, 2, 3, 4]))      # [2, 3, 4, -1]
```

### Example 2: Daily Temperatures

**Problem:** Find how many days until warmer temperature.

```python
def dailyTemperatures(temperatures):
    """
    Time: O(n), Space: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop days with lower temperature
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)
    
    return result

# Example usage
temps = [73, 74, 75, 71, 69, 72, 76, 73]
print(dailyTemperatures(temps))  # [1, 1, 4, 2, 1, 1, 0, 0]
```

### Example 3: Largest Rectangle in Histogram

**Problem:** Find largest rectangular area in histogram.

```python
def largestRectangleArea(heights):
    """
    Time: O(n), Space: O(n)
    """
    stack = []  # Store indices
    max_area = 0
    heights.append(0)  # Sentinel to empty stack
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    heights.pop()  # Remove sentinel
    return max_area

# Example usage
print(largestRectangleArea([2, 1, 5, 6, 2, 3]))  # 10
print(largestRectangleArea([2, 4]))              # 4
```

### Example 4: Trapping Rain Water (Stack Approach)

**Problem:** Calculate trapped rainwater.

```python
def trap(height):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    water = 0
    
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            
            if not stack:
                break
            
            distance = i - stack[-1] - 1
            bounded_height = min(h, height[stack[-1]]) - height[bottom]
            water += distance * bounded_height
        
        stack.append(i)
    
    return water

# Example usage
print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  # 6
```

## Pattern 4: Stack for DFS (Depth-First Search)

### Example 1: Binary Tree Inorder Traversal (Iterative)

**Problem:** Traverse tree inorder without recursion.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorderTraversal(root):
    """
    Time: O(n), Space: O(h) where h is height
    """
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result

# Example usage
#     1
#      \
#       2
#      /
#     3
root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(3)
print(inorderTraversal(root))  # [1, 3, 2]
```

### Example 2: Clone Graph (DFS with Stack)

**Problem:** Deep copy an undirected graph.

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
    """
    Time: O(V + E), Space: O(V)
    """
    if not node:
        return None
    
    clones = {node: Node(node.val)}
    stack = [node]
    
    while stack:
        current = stack.pop()
        
        for neighbor in current.neighbors:
            if neighbor not in clones:
                clones[neighbor] = Node(neighbor.val)
                stack.append(neighbor)
            
            clones[current].neighbors.append(clones[neighbor])
    
    return clones[node]
```

### Example 3: Path Sum (Iterative)

**Problem:** Check if tree has root-to-leaf path with given sum.

```python
def hasPathSum(root, targetSum):
    """
    Time: O(n), Space: O(h)
    """
    if not root:
        return False
    
    stack = [(root, targetSum - root.val)]
    
    while stack:
        node, remaining = stack.pop()
        
        # Leaf node
        if not node.left and not node.right and remaining == 0:
            return True
        
        if node.right:
            stack.append((node.right, remaining - node.right.val))
        if node.left:
            stack.append((node.left, remaining - node.left.val))
    
    return False
```

## Pattern 5: String Manipulation

### Example 1: Remove Adjacent Duplicates

**Problem:** Remove adjacent duplicate characters.

```python
def removeDuplicates(s):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    
    return ''.join(stack)

# Example usage
print(removeDuplicates("abbaca"))  # "ca"
print(removeDuplicates("azxxzy"))  # "ay"
```

### Example 2: Remove K Digits

**Problem:** Remove k digits to get smallest number.

```python
def removeKdigits(num, k):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    
    for digit in num:
        # Remove larger digits
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    
    # Remove remaining k digits from end
    while k > 0:
        stack.pop()
        k -= 1
    
    # Build result, remove leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'

# Example usage
print(removeKdigits("1432219", 3))  # "1219"
print(removeKdigits("10200", 1))    # "200"
```

### Example 3: Simplify Path

**Problem:** Simplify Unix-style file path.

```python
def simplifyPath(path):
    """
    Time: O(n), Space: O(n)
    """
    stack = []
    
    for part in path.split('/'):
        if part == '..' and stack:
            stack.pop()
        elif part and part != '.' and part != '..':
            stack.append(part)
    
    return '/' + '/'.join(stack)

# Example usage
print(simplifyPath("/home/"))           # "/home"
print(simplifyPath("/../"))             # "/"
print(simplifyPath("/home//foo/"))      # "/home/foo"
print(simplifyPath("/a/./b/../../c/"))  # "/c"
```

## Pattern 6: Min/Max Stack

### Example: Min Stack

**Problem:** Design stack with O(1) getMin() operation.

```python
class MinStack:
    """
    All operations: O(1)
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Track minimums
    
    def push(self, val):
        self.stack.append(val)
        # Push current minimum
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]

# Example usage
min_stack = MinStack()
min_stack.push(-2)
min_stack.push(0)
min_stack.push(-3)
print(min_stack.getMin())  # -3
min_stack.pop()
print(min_stack.top())     # 0
print(min_stack.getMin())  # -2
```

### Example: Max Stack

**Problem:** Design stack with O(1) peekMax() and O(log n) popMax().

```python
class MaxStack:
    """
    Push/Pop/Top: O(1)
    PeekMax: O(1)
    PopMax: O(n)
    """
    def __init__(self):
        self.stack = []
        self.max_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.max_stack or val >= self.max_stack[-1]:
            self.max_stack.append(val)
    
    def pop(self):
        val = self.stack.pop()
        if val == self.max_stack[-1]:
            self.max_stack.pop()
        return val
    
    def top(self):
        return self.stack[-1]
    
    def peekMax(self):
        return self.max_stack[-1]
    
    def popMax(self):
        max_val = self.max_stack[-1]
        temp = []
        
        # Pop until we find max
        while self.stack[-1] != max_val:
            temp.append(self.pop())
        
        self.pop()  # Remove max
        
        # Push back other elements
        while temp:
            self.push(temp.pop())
        
        return max_val
```

## Pattern 7: Backtracking with Stack

### Example: Generate Parentheses

**Problem:** Generate all valid combinations of n pairs of parentheses.

```python
def generateParenthesis(n):
    """
    Time: O(4^n / sqrt(n)), Space: O(n)
    """
    result = []
    stack = [("", 0, 0)]  # (current_string, open_count, close_count)
    
    while stack:
        s, open_count, close_count = stack.pop()
        
        if len(s) == 2 * n:
            result.append(s)
            continue
        
        if open_count < n:
            stack.append((s + "(", open_count + 1, close_count))
        
        if close_count < open_count:
            stack.append((s + ")", open_count, close_count + 1))
    
    return result

# Example usage
print(generateParenthesis(3))
# ["((()))", "(()())", "(())()", "()(())", "()()()"]
```

## Common Pitfalls & Tips

### Pitfalls to Avoid:
- **Not checking empty** before pop/peek
- **Using stack.pop(0)** instead of stack.pop() (O(n) vs O(1))
- **Forgetting edge cases** - empty stack, single element
- **Stack overflow** in recursive solutions (use iterative with explicit stack)
- **Not clearing stack** between test cases
- **Wrong order** in DFS (push right before left for left-to-right traversal)

### Pro Tips:
- **Use list as stack** - Most efficient in Python: `stack = []`
- **Check empty first** - Always check before pop/peek
- **Monotonic stack** - For next greater/smaller problems
- **Two stacks** - For min/max tracking or implementing queue
- **Stack for DFS** - Explicit stack replaces recursion
- **Sentinel values** - Add dummy elements to simplify logic
- **Process on pop** - Often easier than processing on push

## Complexity Analysis

### Time Complexity:
| Operation | Array-based | Linked List |
|-----------|-------------|-------------|
| Push | O(1) | O(1) |
| Pop | O(1) | O(1) |
| Peek | O(1) | O(1) |
| Search | O(n) | O(n) |
| isEmpty | O(1) | O(1) |

### Space Complexity:
- **All implementations:** O(n) where n is number of elements

## When to Use Stack

**Use Stack when:**
- ✅ Need LIFO (Last-In-First-Out) ordering
- ✅ Matching/balancing problems (parentheses, brackets)
- ✅ Expression evaluation (postfix, infix)
- ✅ Backtracking problems
- ✅ DFS traversal (iterative)
- ✅ Undo/Redo operations
- ✅ Function call management (call stack)
- ✅ Next greater/smaller element problems

**Don't Use Stack when:**
- ❌ Need FIFO ordering (use Queue)
- ❌ Need random access (use Array/List)
- ❌ Need to access elements other than top
- ❌ BFS traversal (use Queue)

## Stack vs Other Data Structures

| Structure | Access | Insert | Delete | Use Case |
|-----------|--------|--------|--------|----------|
| **Stack** | Top: O(1) | Top: O(1) | Top: O(1) | LIFO, DFS |
| **Queue** | Front: O(1) | Rear: O(1) | Front: O(1) | FIFO, BFS |
| **Array** | Any: O(1) | End: O(1) | End: O(1) | Random access |
| **Linked List** | O(n) | Head: O(1) | Head: O(1) | Dynamic size |

## Problem Categories for Practice

### 1. Balanced Parentheses
- Valid parentheses
- Longest valid parentheses
- Minimum add to make valid
- Remove invalid parentheses

### 2. Expression Evaluation
- Evaluate reverse Polish notation
- Basic calculator I, II, III
- Decode string
- Simplify path

### 3. Monotonic Stack
- Next greater element I, II
- Daily temperatures
- Largest rectangle in histogram
- Trapping rain water

### 4. String Manipulation
- Remove adjacent duplicates
- Remove K digits
- Backspace string compare
- Make the string great

### 5. Tree/Graph Traversal
- Binary tree inorder traversal
- Binary tree preorder traversal
- Binary tree postorder traversal
- Clone graph

### 6. Design Problems
- Min stack
- Max stack
- Implement queue using stacks
- Browser history

### 7. Backtracking
- Generate parentheses
- Letter combinations
- Combination sum
- Subsets

## Quick Implementation Reference

### Basic Stack Operations
```python
# Using list
stack = []
stack.append(1)      # Push
item = stack.pop()   # Pop
top = stack[-1]      # Peek
empty = len(stack) == 0
```

### Stack Class
```python
stack = Stack()
stack.push(1)
item = stack.pop()
top = stack.peek()
empty = stack.is_empty()
```

### Common Patterns
```python
# Monotonic Stack
stack = []
for i, num in enumerate(arr):
    while stack and arr[stack[-1]] < num:
        idx = stack.pop()
        # Process
    stack.append(i)

# Matching Brackets
stack = []
for char in s:
    if char in opening:
        stack.append(char)
    elif stack and matches(stack[-1], char):
        stack.pop()
    else:
        return False
return len(stack) == 0
```

## Summary

Stack is a fundamental data structure with LIFO ordering:

**Key Characteristics:**
- **LIFO principle** - Last in, first out
- **Single access point** - Only top element accessible
- **O(1) operations** - Push, pop, and peek
- **Natural for recursion** - Mimics call stack

**Common Applications:**
1. **Balanced parentheses** - Matching brackets
2. **Expression evaluation** - Calculator, postfix notation
3. **Monotonic stack** - Next greater/smaller elements
4. **DFS traversal** - Iterative depth-first search
5. **Backtracking** - Generate combinations
6. **Undo/Redo** - Text editors, browsers
7. **Function calls** - System call stack

**Key Patterns:**
1. **Basic stack** - Push/pop for LIFO behavior
2. **Monotonic stack** - Maintain increasing/decreasing order
3. **Two stacks** - Track min/max or implement queue
4. **Stack + HashMap** - Frequency tracking
5. **DFS stack** - Replace recursion

**Key Takeaway:** Master stack for expression evaluation, parentheses problems, and monotonic stack patterns. Understanding stack is crucial for DFS, backtracking, and many interview problems!

Practice these patterns and you'll handle stack-based problems with confidence and optimal time complexity!
