---
layout: post
title: "190. Reverse Bits"
tag: leetcode刷题笔记
---

Reverse bits of a given 32 bits unsigned integer.

**Example:**

**Input:** 43261596

**Output:** 964176192

**Explanation:** 43261596 represented in binary as 00000010100101000001111010011100, 
             return 964176192 represented in binary as 00111001011110000010100101000000.

**Follow up:**
If this function is called many times, how would you optimize it?


给定32位无符号整数的反转位。

**例：**

输入：43261596
输出：964176192
说明：43261596以二进制表示为00000010100101000001111010011100，
              以二进制形式返回964176192，表示为00111001011110000010100101000000。

**跟进：**
如果多次调用此函数，您将如何优化它？

**思路：**
十进制转二进制
然后存入列表，然后采用直接反转实现reserve（）


十进制转二进制
~~~
nb = bin(n)
~~~
二进制转十进制
~~~
int('10100111110',2)
~~~

逆序遍历
~~~
for i in range(len(l_nb)-1, -1, -1)
~~~

**网上思路：**
<https://blog.csdn.net/coder_orz/article/details/51705094>

~~~
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        nb = bin(n)[:1:-1]
        print(nb)
        return int(nb + '0'*(32-len(nb)), 2)

if __name__ == "__main__":
    print(Solution().reverseBits(6))
~~~


