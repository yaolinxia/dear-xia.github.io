**Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.**

**Example:**
Given a = 1 and b = 2, return 3.

**Credits:**
Special thanks to @fujiaozhu for adding this problem and creating all test cases.

**思路：**

- 不用加号咋实现？

- 转成数组对每一位进行相加

~~~
class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        while b != 0:
            #c表示与操作
            c = a & b
            #与的操作左移，与下一个异或结果，再进行异或操作
            a = (a ^ b) % 0x100000000
            print(a)
            #进位不断进行左移
            b = (c << 1) % 0x100000000
        return a if a <= 0x7FFFFFFF else a | (~0x100000000+1)

if __name__ == "__main__":
    print(Solution().getSum(5, 7))
~~~



**网上思路：**

- 位置不一样结果为1 
- 计算机组成原理加法器

<https://www.polarxiong.com/archives/LeetCode-371-sum-of-two-integers.html>
<https://blog.csdn.net/kekong0713/article/details/52805959>

~~~
1. 求两个整数之和，不能使用加减乘除法。
分析：
不能采用常规的方法来求解的话，可以采用位运算的方法来求和。可以将加法分为三个步骤：第一步，各位相加但不进位，第二步，有进位的话进行进位，第三步，将前面的结果相加。整数在计算机内部的表示本质就是二进制形式，不考虑进位的逐位相加可以用异或运算来表示，对于判断两位相加的结果是否产生进位，可以采用与运算，只有当对应位上的数字均为1时才产生进位，相与为1，其他情况都是0，产生进位，需要向左移动一位，最后将上面两个结果相加，不断重复这个过程，知道不产生进位。
例子：
5的二进制表示是：101  记为n1
7的二进制表示：111  记为n2
相异或运算：010  记为s1
相与运算：101，左移一位，1010， 记为 c1
相加：s1^c1, 1000,
          s1&c1,0010,左移一位，0100，
相加:  s1^c1, 1100， 最终结果
          s1&c1,0  无进位。
~~~

<https://www.hrwhisper.me/leetcode-sum-two-integers/>
~~~
做这题要保证两个数在正确的范围内（本题是int，32bit）

如何做到呢？我们知道32bit 可以表示的无符号整数位0~0xFFFFFFFF（全0~全1）

因此，我们使用&来保证该数是32bit.

int的0和正整数范围为0~0x7FFFFFFF，int负数的范围为-0x80000000~-1,因此，大于0x7FFFFFFF的其实是最高位为1（这是符号位）。这样算出来是把最高位不当成符号位，我们还需要对负数的情况进行修正。

在具体实现上，我们可以先 &0x7FFFFFFF 然后取反，这样，-1变为-0x80000000(-2147483648) -2变为了-0x7FFFFFFF(-2147483647) ,因此，在^0x7FFFFFFF即可。。

class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        MOD     = 0xFFFFFFFF
        MAX_INT = 0x7FFFFFFF
        while b != 0:
            a, b = (a ^ b) & MOD, ((a & b) << 1) & MOD
        return a if a <= MAX_INT else ~(a & MAX_INT) ^ MAX_INT
~~~

>按位取反操作：<https://blog.csdn.net/xiexievv/article/details/8124108>
><https://blog.csdn.net/pipisorry/article/details/36517411>

**存在问题:**

- 负数表示那边不是特别清楚
- 参考网址：<http://www.cnblogs.com/zhangziqiu/archive/2011/03/30/ComputerCode.html>