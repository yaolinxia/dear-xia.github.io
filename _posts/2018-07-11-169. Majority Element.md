Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.

You may assume that the array is non-empty and the majority element always exist in the array.

**Example 1:**

Input: [3,2,3]

Output: 3

**Example 2:**

Input: [2,2,1,1,1,2,2]

Output: 2

给定大小为n的数组，找到多数元素。 多数元素是出现超过⌊n /2⌋倍的元素。

您可以假设该数组非空，并且多数元素始终存在于数组中。

例1：

输入：[3,2,3]

输出：3

例2：

输入：[2,2,1,1,1,2,2]
输出：2

**思路：**
遍历数组，
定义临时存储，
蛮力法

代码如下：
~~~
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length == 1:
            return nums[0]
        sublen = int(length/2)
        #最多元素的出现次数
        maxNum = 0
        #出现最多的元素
        maxN = 0
        for i in range(0, length):
            sumNums = 0
            for j in range(i+1, length):
                if nums[j] == nums[i]:
                    sumNums += 1
            if sumNums >= sublen and sumNums > maxNum:
                maxNum = sumNums
                maxN = nums[i]
        return maxN

if __name__ == "__main__":
    nums = [2]
    print(Solution().majorityElement(nums))
~~~
>出现问题：提交时代码时间超出 Time Limit Exceeded

**思路2：**
先进行排序，找出中间元素，因为个数超过一半，所以中间元素应该就是最后的值
>排序，直接使用sort()
~~~
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        length = len(nums)
        if length == 1:
            return nums[0]
        sublen = int(length/2)
        return nums[sublen]

if __name__ == "__main__":
    nums = [2,2,2,3,3,4]
    print(Solution().majorityElement(nums))
~~~
**网上思路：**
>网上思路很多，参见<https://blog.csdn.net/coder_orz/article/details/51407713>

