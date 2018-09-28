---
layout: post
title: "206. Reverse Linked List"
tag: leetcode刷题笔记
---

Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
Follow up:

A linked list can be reversed either iteratively or recursively. Could you implement both?
链接列表可以反复或递归地反转。 你能同时实施吗？

**思路：**
有点类似于栈的思想
先进后出

~~~
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        stack = []
        i = 0
        p = head
        while p:
            stack.append(p.val)
            #print(p.val)
            p = p.next
            i += 1
        #print(stack)
        q = head
        l = []
        for j in range(len(stack)-1, -1, -1):
            #print(j)
            #q.val = stack[j]
            l.append(stack[j])
            #print(q.val)
            #q = q.next
        #print(l)
        return l


if __name__ == "__main__":
    L1 = ListNode(1)
    L1.next = ListNode(2)
    L1.next.next = ListNode(3)
    L1.next.next.next = ListNode(4)
    L1.next.next.next.next = ListNode(5)
    Solution().reverseList(L1)
~~~
>通过，输出的形式必须是数组，才会通过

**网上思路：**
<https://blog.csdn.net/coder_orz/article/details/51306170>

~~~
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        p = head
        newList = []
        while p:
            newList.insert(0, p.val)
            p = p.next

        p = head
        for v in newList:
            p.val = v
            print(p.val)
            p = p.next
        return head



if __name__ == "__main__":
    L1 = ListNode(1)
    L1.next = ListNode(2)
    L1.next.next = ListNode(3)
    L1.next.next.next = ListNode(4)
    L1.next.next.next.next = ListNode(5)
    print(Solution().reverseList(L1))
~~~
疑问：为啥是return head

