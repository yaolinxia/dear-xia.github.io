---
layout: post
title: "3. Longest Substring Without Repeating Characters"
tag: leetcode刷题笔记
---

Given a string, find the length of the longest substring without repeating characters.

**Examples:**

Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

给定一个字符串，找到最长子字符串的长度而不重复字符。

**例子：**

给定“abcabcbb”，答案是“abc”，长度为3。

给定“bbbbb”，答案是“b”，长度为1。

给定“pwwkew”，答案是“wke”，长度为3.注意答案必须是子字符串，“pwke”是子序列而不是子字符串。

**网上思路：**
<https://blog.csdn.net/daigualu/article/details/73838465>

~~~python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        maxLen = 0
        curmax = 0
        subStr = ""
        #索引
        i = 0
        length = len(s)
        while i < length:
            #找到当前字符的位置,看subStr中是否存在
            ind = subStr.find(s[i])
            if ind == -1:
                subStr+=s[i]
                curmax += 1

            else:
                if maxLen < curmax:
                    maxLen = curmax
                i = i - len(subStr) + ind
                subStr = ""
                curmax = 0
            i += 1
        if maxLen < curmax:
            maxLen = curmax
        return maxLen

if __name__ == "__main__":
    s1 = "abccc"
    print(Solution().lengthOfLongestSubstring(s1))
~~~
