---
layout: post
title: "204. Count Primes"
tag: leetcode刷题笔记
---
Count the number of prime numbers less than a non-negative number, n.
计算小于非负数n的素数的数量。
**Example:**

**Input:** 10

**Output:** 4

Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.

**思路：**
1. 计算素数的方法
2. 看是否有可以被其整除的数
3. 在1-n之间进行遍历，找出素数

> 蛮力法，会超时;采用网上思路：厄拉多塞筛法

网上思路：<https://blog.csdn.net/coder_orz/article/details/51321944>
<https://blog.csdn.net/github_39261590/article/details/73864039>
