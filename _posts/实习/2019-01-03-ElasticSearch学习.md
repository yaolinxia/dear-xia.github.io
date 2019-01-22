---
layout: post
title: "ElasticSearch学习"
tag: 实习
---

[TOC]

# 基本概念

## Near Realtime(NRT)
Elasticsearch是一个近实时搜索平台。 这意味着从索引文档到可搜索文档的时间有一点延迟（通常是一秒）。

## Cluster
A cluster is a collection of one or more nodes (servers) that together holds your entire data and provides federated indexing and search capabilities across all nodes. 
集群是一个或多个节点（服务器）的集合，它们共同保存您的整个数据，并提供跨所有节点的联合索引和搜索功能。

A cluster is identified by a unique name which by default is "elasticsearch". This name is important because a node can only be part of a cluster if the node is set up to join the cluster by its name.
集群由唯一名称标识，默认情况下为“elasticsearch”。 此名称很重要，因为如果节点设置为按名称加入集群，则该节点只能是集群的一部分。
## Node
一个单独的服务器，是你的集群的一部分，存储你的数据，并且参与到集群的索引以及，搜索。
一个节点可以加入一个特定的集群，通过集群的名称，默认节点都会被加入到一个叫做elasticsearch的集群中。

## Index
An index is a collection of documents that have somewhat similar characteristics.
索引是具有某些类似特征的文档集合。

 For example, you can have an index for customer data, another index for a product catalog, and yet another index for order data. An index is identified by a name (that must be all lowercase) and this name is used to refer to the index when performing indexing, search, update, and delete operations against the documents in it.

In a single cluster, you can define as many indexes as you want.
例如，您可以拥有客户数据的索引，产品目录的另一个索引以及订单数据的另一个索引。 索引由名称标识（必须全部小写），此名称用于在对其中的文档执行索引，搜索，更新和删除操作时引用索引。

在单个群集中，您可以根据需要定义任意数量的索引。

## type
在6.0.0中被弃用
A type used to be a logical category/partition of your index to allow you to store different types of documents in the same index, e.g. one type for users, another type for blog posts. It is no longer possible to create multiple types in an index, and the whole concept of types will be removed in a later version. See Removal of mapping types for more.
一种类型，曾经是索引的逻辑类别/分区，允许您在同一索引中存储不同类型的文档，例如 一种用户类型，另一种用于博客帖子。 不再可能在索引中创建多个类型，并且将在更高版本中删除类型的整个概念。 请参阅删除映射类型以获取更多信息。

## Document
信息的基本单元，json格式


## Shards&Replicas
解决索引存储大容量数据，过慢的问题
To solve this problem, Elasticsearch provides the ability to subdivide your index into multiple pieces called shards. When you create an index, you can simply define the number of shards that you want. Each shard is in itself a fully-functional and independent "index" that can be hosted on any node in the cluster.
为了解决这个问题，Elasticsearch提供了将索引细分为多个称为分片的功能。 创建索引时，只需定义所需的分片数即可。 每个分片本身都是一个功能齐全且独立的“索引”，可以托管在集群中的任何节点上。
Sharding is important for two primary reasons:

It allows you to horizontally split/scale your content volume

It allows you to distribute and parallelize operations across shards (potentially on multiple nodes) thus increasing performance/throughput
分片很重要，主要有两个原因：

它允许您水平拆分/缩放内容量
它允许您跨分片（可能在多个节点上）分布和并行化操作，从而提高性能/吞吐量

The mechanics of how a shard is distributed and also how its documents are aggregated back into search requests are completely managed by Elasticsearch and is transparent to you as the user.
分片的分布方式以及如何将其文档聚合回搜索请求的机制完全由Elasticsearch管理，对用户而言是透明的。

为了避免分片或者节点的消失，elasticsearch允许复制分片
Replication is important for two primary reasons:

It provides high availability in case a shard/node fails. For this reason, it is important to note that a replica shard is never allocated on the same node as the original/primary shard that it was copied from.

It allows you to scale out your search volume/throughput since searches can be executed on all replicas in parallel.

复制很重要，主要有两个原因：
它在碎片/节点发生故障时提供高可用性。 因此，请务必注意，副本分片永远不会在与从中复制的原始/主分片相同的节点上分配。
它允许您扩展搜索量/吞吐量，因为可以在所有副本上并行执行搜索。

To summarize, each index can be split into multiple shards. An index can also be replicated zero (meaning no replicas) or more times. Once replicated, each index will have primary shards (the original shards that were replicated from) and replica shards (the copies of the primary shards).
总而言之，每个索引可以拆分为多个分片。 索引也可以复制为零（表示没有副本）或更多次。 复制后，每个索引都将具有主分片（从中复制的原始分片）和副本分片（主分片的副本）。

The number of shards and replicas can be defined per index at the time the index is created. After the index is created, you may also change the number of replicas dynamically anytime. You can change the number of shards for an existing index using the _shrink and _split APIs, however this is not a trivial task and pre-planning for the correct number of shards is the optimal approach.

可以在创建索引时为每个索引定义分片和副本的数量。 创建索引后，您还可以随时动态更改副本数。 您可以使用_shrink和_split API更改现有索引的分片数，但这不是一项简单的任务，预先计划正确数量的分片是最佳方法。

By default, each index in Elasticsearch is allocated 5 primary shards and 1 replica which means that if you have at least two nodes in your cluster, your index will have 5 primary shards and another 5 replica shards (1 complete replica) for a total of 10 shards per index.

默认情况下，Elasticsearch中的每个索引都分配了5个主分片和1个副本，这意味着如果群集中至少有两个节点，则索引将包含5个主分片和另外5个副本分片（1个完整副本），总计为 每个索引10个分片。


Each Elasticsearch shard is a Lucene index. There is a maximum number of documents you can have in a single Lucene index. As of LUCENE-5843, the limit is 2,147,483,519 (= Integer.MAX_VALUE - 128) documents. You can monitor shard sizes using the _cat/shards API.

每个Elasticsearch分片都是Lucene索引。 单个Lucene索引中可以包含最大数量的文档。 自LUCENE-5843起，限制为2,147,483,519（= Integer.MAX_VALUE  -  128）个文件。 您可以使用_cat / shards API监视分片大小。
![image.png](http://note.youdao.com/yws/res/9467/WEBRESOURCE3e9feef2f7d83923c99b773c0fc55247)

# Exploring Your cluster探索集群
## The REST API
如上所示，我们已经有了集群，节点。下一步就是理解如何与它进行交流。
Elasticsearch提供了强大的REST API可以与你构建的集群进行交互
该API可以做一下几件事
- 检查集群，节点，索引健康，状态，统计数据
- 管理你的集群，节点，索引数据以及原子数据
- 执行创建，读取，更新，删除等操作
- 执行高级搜索操作，例如分页，排序，过滤，脚本编写，聚合等等

## Cluster Health 
从基本的集群检查开始，查看集群正在做什么。
To check the cluster health, we will be using the _cat API. You can run the command below in Kibana’s Console by clicking "VIEW IN CONSOLE" or with curl by clicking the "COPY AS CURL" link below and pasting it into a terminal.
要检查群集运行状况，我们将使用_cat API。 您可以通过单击“查看控制台”或通过单击下面的“COPY AS CURL”链接并将其粘贴到终端中，在Kibana控制台中运行以下命令。

Whenever we ask for the cluster health, we either get green, yellow, or red.

Green - everything is good (cluster is fully functional)

Yellow - all data is available but some replicas are not yet allocated (cluster is fully functional)

Red - some data is not available for whatever reason (cluster is partially functional)
每当我们要求群集健康时，我们要么获得绿色，黄色或红色。

绿色 - 一切都很好（集群功能齐全）

黄色 - 所有数据均可用，但尚未分配一些副本（群集功能齐全）

红色 - 某些数据由于某种原因不可用（群集部分功能）

Note: When a cluster is red, it will continue to serve search requests from the available shards but you will likely need to fix it ASAP since there are unassigned shards.

注意：当群集为红色时，它将继续提供来自可用分片的搜索请求，但您可能需要尽快修复它，因为存在未分配的分片。

Also from the above response, we can see a total of 1 node and that we have 0 shards since we have no data in it yet. Note that since we are using the default cluster name (elasticsearch) and since Elasticsearch uses unicast network discovery by default to find other nodes on the same machine, it is possible that you could accidentally start up more than one node on your computer and have them all join a single cluster. In this scenario, you may see more than 1 node in the above response.

同样从上面的响应中，我们可以看到总共1个节点，并且我们有0个分片，因为我们还没有数据。 请注意，由于我们使用的是默认群集名称（elasticsearch），并且由于Elasticsearch默认使用单播网络发现来查找同一台计算机上的其他节点，因此您可能会意外启动计算机上的多个节点并拥有它们 所有人都加入一个集群。 在这种情况下，您可能会在上面的响应中看到多个节点。

~~~
curl -X GET "localhost:9200/_cat/health?v"
GET /_cat/health?v
~~~



## List All Indices 
~~~
GET /_cat/indices?v
curl -X GET "localhost:9200/_cat/indices?v"
~~~



## create an Index创建一个索引
创建一个叫做customer的索引
~~~
curl -X PUT "localhost:9200/customer?pretty"
~~~
该命令创建了一个叫做customer的索引
We simply append pretty to the end of the call to tell it to pretty-print the JSON response (if any).
我们只是简单地追加到调用的结尾，告诉它打印JSON响应（如果有的话）。
![](E:\yaolinxia\workspace\practice\practice\images\create_index.png)


## Index and Query a Document 
现在开始添加东西进入customer索引中去，
We’ll index a simple customer document into the customer index, with an ID of 1 as follows:

我们将一个简单的客户文档索引到客户索引中，ID为1，如下所示：
~~~
curl -X PUT "localhost:9200/customer/_doc/1?pretty" -H 'Content-Type: application/json' -d'
{
  "name": "John Doe"
}
'
~~~
**结果**
~~~
{
  "_index" : "customer",
  "_type" : "_doc",
  "_id" : "1",
  "_version" : 1,
  "result" : "created",
  "_shards" : {
    "total" : 2,
    "successful" : 1,
    "failed" : 0
  },
  "_seq_no" : 0,
  "_primary_term" : 1
}
~~~
## Delete an Index
~~~
curl -X DELETE "localhost:9200/customer?pretty"
curl -X GET "localhost:9200/_cat/indices?v"
~~~
**结果**
~~~
health status index uuid pri rep docs.count docs.deleted store.size pri.store.size
~~~
![image.png](http://note.youdao.com/yws/res/9553/WEBRESOURCEd81b72dceda19f8b4611ef24186f779e)

If we study the above commands carefully, we can actually see a pattern of how we access data in Elasticsearch. That pattern can be summarized as follows:
如果我们仔细研究上述命令，我们实际上可以看到我们如何在Elasticsearch中访问数据的模式。 该模式可归纳如下：
~~~
<HTTP Verb> /<Index>/<Type>/<ID>
~~~

## Modifying data
时效性很强
Elasticsearch provides data manipulation and search capabilities in near real time. By default, you can expect a one second delay (refresh interval) from the time you index/update/delete your data until the time that it appears in your search results. This is an important distinction from other platforms like SQL wherein data is immediately available after a transaction is completed.\
Elasticsearch几乎实时提供数据操作和搜索功能。 默认情况下，从索引/更新/删除数据到搜索结果中显示的时间，您可能会有一秒钟的延迟（刷新间隔）。 这是与SQL等其他平台的重要区别，其中数据在事务完成后立即可用。

### 索引、替换记录

![image.png](http://note.youdao.com/yws/res/9567/WEBRESOURCE4ab0c7afa475f615e47d2bb1d07c5f92)
 **如果没有一个具体的ID**
 使用POST
 ~~~
 curl -X POST "localhost:9200/customer/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "name": "Jane Doe"
}
'

 ~~~

## Updating Documents
In addition to being able to index and replace documents, we can also update documents. Note though that Elasticsearch does not actually do in-place updates under the hood. Whenever we do an update, Elasticsearch deletes the old document and then indexes a new document with the update applied to it in one shot.
除了能够索引和替换文档，我们还可以更新文档。 请注意，Elasticsearch实际上并没有在内部进行就地更新。 每当我们进行更新时，Elasticsearch都会删除旧文档，然后一次性对应用了更新的新文档编制索引。
![](E:\yaolinxia\workspace\practice\practice\images\微信截图_20190103174554.png)

![微信截图_20190103174616](E:\yaolinxia\workspace\practice\practice\images\微信截图_20190103174616.png)

![微信截图_20190103174623](E:\yaolinxia\workspace\practice\practice\images\微信截图_20190103174623.png)

n the above example, ctx._source refers to the current source document that is about to be updated.

Elasticsearch provides the ability to update multiple documents given a query condition (like an SQL UPDATE-WHERE statement). See docs-update-by-query API

在上面的示例中，ctx._source指的是即将更新的当前源文档。
Elasticsearch提供了在给定查询条件（如SQL UPDATE-WHERE语句）的情况下更新多个文档的功能。 请参阅docs-update-by-query API

## Delete Documents
![](E:\yaolinxia\workspace\practice\practice\images\微信截图_20190103174654.png)


## Batch Processing
In addition to being able to index, update, and delete individual documents, Elasticsearch also provides the ability to perform any of the above operations in batches using the _bulk API. This functionality is important in that it provides a very efficient mechanism to do multiple operations as fast as possible with as few network roundtrips as possible.

除了能够索引，更新和删除单个文档之外，Elasticsearch还提供了使用_bulk API批量执行上述任何操作的功能。 此功能非常重要，因为它提供了一种非常有效的机制，可以尽可能快地执行多个操作，并尽可能少地进行网络往返。

![](E:\yaolinxia\workspace\practice\practice\images\image (1).png)

![image](E:\yaolinxia\workspace\practice\practice\images\image.png)

# Exploring your data 

Now that we’ve gotten a glimpse of the basics, let’s try to work on a more realistic dataset. I’ve prepared a sample of fictitious JSON documents of customer bank account information. Each document has the following schema:
现在我们已经了解了基础知识，让我们尝试更真实的数据集。 我准备了一份关于客户银行账户信息的虚构JSON文档样本。 每个文档都有以下架构：

~~~
{
    "account_number": 0,
    "balance": 16623,
    "firstname": "Bradshaw",
    "lastname": "Mckenzie",
    "age": 29,
    "gender": "F",
    "address": "244 Columbus Place",
    "employer": "Euron",
    "email": "bradshawmckenzie@euron.com",
    "city": "Hobucken",
    "state": "CO"
}
~~~
For the curious, this data was generated using www.json-generator.com/, so please ignore the actual values and semantics of the data as these are all randomly generated.
奇怪的是，这些数据是使用www.json-generator.com/生成的，因此请忽略数据的实际值和语义，因为这些都是随机生成的。

### 下载样本数据集
![](E:\yaolinxia\workspace\practice\practice\images\image (2).png)

## The Search API

**有两种方法**
- REST 请求URI
- REST 请求body

Now let’s start with some simple searches. There are two basic ways to run searches: one is by sending search parameters through the REST request URI and the other by sending them through the REST request body. 
现在让我们从一些简单的搜索开始吧。 运行搜索有两种基本方法：一种是通过REST请求URI发送搜索参数，另一种是通过REST请求体发送搜索参数。

请求体方法允许您更具表现力，并以更易读的JSON格式定义搜索。
![image.png](http://note.youdao.com/yws/res/9627/WEBRESOURCE46dcdcee158a2f036c822e7d9eb40135)
Let’s first dissect the search call. We are searching (_search endpoint) in the bank index, and the q=* parameter instructs Elasticsearch to match all documents in the index. The sort=account_number:asc parameter indicates to sort the results using the account_number field of each document in an ascending order. The pretty parameter, again, just tells Elasticsearch to return pretty-printed JSON results.

让我们首先剖析搜索电话。 我们正在银行索引中搜索（_search endpoint），q = *参数指示Elasticsearch匹配索引中的所有文档。 sort = account_number：asc参数指示使用升序中的每个文档的account_number字段对结果进行排序。 漂亮的参数再次告诉Elasticsearch返回漂亮的JSON结果。
As for the response, we see the following parts:

- took – time in milliseconds for Elasticsearch to execute the search(以毫秒为单位)
- timed_out – tells us if the search timed out or not(是否超时)
- _shards – tells us how many shards were searched, as well as a count of the successful/failed searched shards(有多少分片)
- hits – search results
- hits.total – total number of documents matching our search criteria(符合搜索条件的文档总数)
- hits.hits – actual array of search results (defaults to first 10 documents)(搜索结果的实际数组)
- hits.sort - sort key for results (missing if sorting by score)(结果的排序键)
- hits._score and max_score - ignore these fields for now
(暂时忽略这些字段)

It is important to understand that once you get your search results back, Elasticsearch is completely done with the request and does not maintain any kind of server-side resources or open cursors into your results. This is in stark contrast to many other platforms such as SQL wherein you may initially get a partial subset of your query results up-front and then you have to continuously go back to the server if you want to fetch (or page through) the rest of the results using some kind of stateful server-side cursor.

重要的是要理解，一旦您获得了搜索结果，Elasticsearch就完全完成了请求，并且不会在结果中维护任何类型的服务器端资源或打开游标。 这与SQL等许多其他平台形成鲜明对比，其中您可能最初预先获得查询结果的部分子集，然后如果要获取（或翻页）其余的则必须连续返回服务器 使用某种有状态服务器端游标的结果。

### Introducing the Query Language介绍查询语言
![](E:\yaolinxia\workspace\practice\practice\images\image (3).png)

### Executing Searchs
不获取整个document
Now that we have seen a few of the basic search parameters, let’s dig in some more into the Query DSL. Let’s first take a look at the returned document fields. By default, the full JSON document is returned as part of all searches. This is referred to as the source (_source field in the search hits). If we don’t want the entire source document returned, we have the ability to request only a few fields from within source to be returned.
![image.png](E:\yaolinxia\workspace\practice\practice\images\image (4).png)
如上图所示，只返回source中的两个参数

#### match query
可以看作是一个基本的领域搜索询问
![image.png](E:\yaolinxia\workspace\practice\practice\images\image (5).png)

![image.png](E:\yaolinxia\workspace\practice\practice\images\image (6).png)

![image.png](E:\yaolinxia\workspace\practice\practice\images\image (7).png)

#### bool query
and
![image.png](E:\yaolinxia\workspace\practice\practice\images\image (8).png)

or
![image.png](E:\yaolinxia\workspace\practice\practice\images\image (9).png)

In the above example, the bool should clause specifies a list of queries either of which must be true for a document to be considered a match.

This example composes two match queries and returns all accounts that contain neither "mill" nor "lane" in the address:
在上面的示例中，bool should子句指定了一个查询列表，其中任何一个查询都必须为true才能将文档视为匹配项。

此示例组成两个匹配查询，并返回地址中既不包含“mill”也不包含“lane”的所有帐户：
![image.png](E:\yaolinxia\workspace\practice\practice\images\image (10).png)

![image.png](E:\yaolinxia\workspace\practice\practice\images\image (11).png)

## Executing Filters
In the previous section, we skipped over a little detail called the document score (_score field in the search results). The score is a numeric value that is a relative measure of how well the document matches the search query that we specified. The higher the score, the more relevant the document is, the lower the score, the less relevant the document is.
在上一节中，我们跳过了一个称为文档分数的小细节（搜索结果中的_score字段）。 分数是一个数值，它是文档与我们指定的搜索查询匹配程度的相对度量。 分数越高，文档越相关，分数越低，文档的相关性越低。

But queries do not always need to produce scores, in particular when they are only used for "filtering" the document set. Elasticsearch detects these situations and automatically optimizes query execution in order not to compute useless scores.
但是查询并不总是需要产生分数，特别是当它们仅用于“过滤”文档集时。 Elasticsearch会检测这些情况并自动优化查询执行，以便不计算无用的分数。

The bool query that we introduced in the previous section also supports filter clauses which allow us to use a query to restrict the documents that will be matched by other clauses, without changing how scores are computed. 
我们在上一节中介绍的bool查询还支持过滤子句，这些子句允许我们使用查询来限制将与其他子句匹配的文档，而不会更改计算得分的方式。
As an example, let’s introduce the range query, which allows us to filter documents by a range of values. This is generally used for numeric or date filtering.
作为一个例子，让我们介绍范围查询，它允许我们按一系列值过滤文档。 这通常用于数字或日期过滤。
This example uses a bool query to return all accounts with balances between 20000 and 30000, inclusive. In other words, we want to find accounts with a balance that is greater than or equal to 20000 and less than or equal to 30000.
![image.png](http://note.youdao.com/yws/res/9694/WEBRESOURCE374c3dcf35a898260b3a8497b1b3bd7e)
Dissecting the above, the bool query contains a match_all query (the query part) and a range query (the filter part). We can substitute any other queries into the query and the filter parts. In the above case, the range query makes perfect sense since documents falling into the range all match "equally", i.e., no document is more relevant than another.

解析上面的内容，bool查询包含match_all查询（查询部分）和范围查询（过滤器部分）。 我们可以将任何其他查询替换为查询和过滤器部分。 在上述情况下，范围查询非常有意义，因为落入范围的文档都“相同”匹配，即，没有文档比另一文档更相关。
In addition to the match_all, match, bool, and range queries, there are a lot of other query types that are available and we won’t go into them here. Since we already have a basic understanding of how they work, it shouldn’t be too difficult to apply this knowledge in learning and experimenting with the other query types.
除了match_all，match，bool和range查询之外，还有很多其他可用的查询类型，我们不会在这里讨论它们。 由于我们已经基本了解它们的工作原理，因此将这些知识应用于学习和试验其他查询类型应该不会太困难。

## Executing Aggregations
Aggregations provide the ability to group and extract statistics from your data. The easiest way to think about aggregations is by roughly equating it to the SQL GROUP BY and the SQL aggregate functions. In Elasticsearch, you have the ability to execute searches returning hits and at the same time return aggregated results separate from the hits all in one response. 
聚合提供了从数据中分组和提取统计信息的功能。 考虑聚合的最简单方法是将其大致等同于SQL GROUP BY和SQL聚合函数。 在Elasticsearch中，您可以执行返回匹配的搜索，同时在一个响应中返回与命中相关的聚合结果。
![image.png](http://note.youdao.com/yws/res/9705/WEBRESOURCEa78f34260e5f35cce52a5ec2241cabfe)

We can see that there are 27 accounts in ID (Idaho), followed by 27 accounts in TX (Texas), followed by 25 accounts in AL (Alabama), and so forth.

Note that we set size=0 to not show search hits because we only want to see the aggregation results in the response.

Building on the previous aggregation, this example calculates the average account balance by state (again only for the top 10 states sorted by count in descending order):
我们可以看到ID（爱达荷州）有27个账户，其次是TX（德克萨斯州）的27个账户，其次是AL（阿拉巴马州）的25个账户，依此类推。

请注意，我们将size = 0设置为不显示搜索命中，因为我们只想在响应中看到聚合结果。

在前一个聚合的基础上，此示例按州计算平均帐户余额（同样仅针对按降序排序的前10个州）：
![image.png](E:\yaolinxia\workspace\practice\practice\images\image (12).png)

![image.png](E:\yaolinxia\workspace\practice\practice\images\image (13).png)

This example demonstrates how we can group by age brackets (ages 20-29, 30-39, and 40-49), then by gender, and then finally get the average account balance, per age bracket, per gender:
此示例演示了我们如何按年龄段（20-29岁，30-39岁和40-49岁）进行分组，然后按性别分组，最后得到每个年龄段的平均帐户余额：
~~~
curl -X GET "localhost:9200/bank/_search" -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "aggs": {
    "group_by_age": {
      "range": {
        "field": "age",
        "ranges": [
          {
            "from": 20,
            "to": 30
          },
          {
            "from": 30,
            "to": 40
          },
          {
            "from": 40,
            "to": 50
          }
        ]
      },
      "aggs": {
        "group_by_gender": {
          "terms": {
            "field": "gender.keyword"
          },
          "aggs": {
            "average_balance": {
              "avg": {
                "field": "balance"
              }
            }
          }
        }
      }
    }
  }
}
'
~~~

![image.png](E:\yaolinxia\workspace\practice\practice\images\image (14).png)



![](E:\yaolinxia\workspace\practice\practice\images\image (15).png)

![image (16)](E:\yaolinxia\workspace\practice\practice\images\image (16).png)

![image (17)](E:\yaolinxia\workspace\practice\practice\images\image (17).png)

# 参考网址

- <https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started-cluster-health.html>


