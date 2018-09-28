---
layout: post
title: "217. Contains Duplicate"
tag: leetcode刷题笔记
---

Given an array of integers, find if the array contains any duplicates.

Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

**Example 1:**

**Input:** [1,2,3,1]

**Output:** true

**Example 2:**

**Input:** [1,2,3,4]

**Output:** false

**Example 3:**

**Input:** [1,1,1,3,3,4,3,2,4,2]

**Output:** true

**思路：**
1.先排序
2.看最大值与最小值的差与列表长度进行比较

**代码有问题**
~~~
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        length = len(nums)
        if length <= 1:
            return False
        nums.sort()
        print(nums)
        minS = min(nums)
        maxS = max(nums)
        if (maxS-minS) < length-1:
            return True
        else:
            return False

if __name__ == "__main__":
    nums = [1]
    b = Solution().containsDuplicate(nums)
    print(b)
~~~

**网上思路：**<https://blog.csdn.net/coder_orz/article/details/51407597>
~~~
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        map = {}
        for i in nums:
            print(i)
            if i in map:
                #print(map([i]))
                return True
            map[i] = True
        return False

if __name__ == "__main__":
    nums = [1,1,1,3,3,4,3,2,4,2]
    b = Solution().containsDuplicate(nums)
    print(b)
~~~
