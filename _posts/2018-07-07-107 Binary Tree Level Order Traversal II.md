---
layout: post
title: "107 binary tree level order traversal ii"
tag: leetcode刷题笔记
---
Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its bottom-up level order traversal as:
[
  [15,7],
  [9,20],
  [3]
]

给定一个二叉树，返回其节点值的自底向上的级别遍历。 （即从左到右，从叶到根逐级）。

例如：
给定二叉树[3,9,20，null，null，15,7]，
    3
    / \
   9 20
     / \
    15 7
返回其自下而上的级别遍历，如下所示：
[
  [15,7]，
  [9,20]，
  [3]
]

>思路：借鉴网上思路，<https://shenjie1993.gitbooks.io/leetcode-python/107%20Binary%20Tree%20Level%20Order%20Traversal%20II.html>
>首先定义空列表res,将树每一层的节点存在一个列表中，遍历列表中的元素，如果该节点有左右节点的话，就把它们加入一个临时列表，这样当遍历结束时，下一层的节点也按照顺序存储好了，不断循环直到下一层的列表为空。


~~~
# Definition for a binary tree node.
from datashape import null


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        #存放最终结果
        res = []
        #如果为空，返回结果列表
        if not root:
            return res
        curr_root = [root]
        while curr_root:
            level_result = []
            next_root = []
            for temp in curr_root:
                level_result.append(temp.val)
                if temp.left:
                    next_root.append(temp.left)
                if temp.right:
                    next_root.append(temp.right)
            res.append(level_result)
            curr_root = next_root
        res.reverse()
        return res

if __name__ == "__main__":
    t = TreeNode(3)
    t.left = TreeNode(9)
    t.right = TreeNode(20)
    t.left.left = TreeNode("null")
    t.left.right = TreeNode("null")
    t.right.left = TreeNode(15)
    t.right.right = TreeNode(7)

    print(Solution().levelOrderBottom(t))
    #print(Solution().levelOrderBottom(t))




~~~
