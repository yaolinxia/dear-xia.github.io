---
layout: post
title: "191. Number of 1 Bits"
tag: leetcode刷题笔记
---
Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

**Example 1:**

**Input:** 11

**Output:** 3

**Explanation:** Integer 11 has binary representation 00000000000000000000000000001011 

**Example 2:**

**Input:** 128

**Output:** 1

**Explanation:** Integer 128 has binary representation 00000000000000000000000010000000

**思路：**
1. 转化为二进制数，直接遍历
2. 蛮力法
~~~
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        nb = bin(n)[2::1]
        print(nb)
        c = ('0'*(32-len(nb)) + nb)
        print(c)
        i = 0
        for temp in c:
            if temp == '1':
                i += 1
        return i


if __name__ == "__main__":
    n = Solution().hammingWeight(128)
    print(n)
~~~

**网上思路：**
<https://blog.csdn.net/coder_orz/article/details/51323188>
