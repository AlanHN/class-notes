# Advanced Distributed System

## 并行与分布式系统（lec1）

- 现有系统大多数都是并行与分布式系统，都是MIMD架构的
- 定义：一组独立的机子，通过网络连接，协同工作向外提供服务
- 限制：没有共享的memory或者clock
- 考虑的主要问题与特性
  - 可靠性：可用性与持久性
  - 性能：主要考虑网络速度比较慢
  - 可扩展性：考虑资源数目、地理条件（时延、带宽、可靠性）、复杂度
  - 透明性：用户无感知（高层次）、程序无感知（低层次）
- 主要模型
  - 中心模型：就是传统机子
  - Client-server模型：client请求，server干活
  - 点对点模型：所有人即是client，也是server
  - Processor Pool 模型：一堆资源
  - 云计算 Model
    - IaaS, PaaS, SaaS, FaaS（例子看ppt）
- Case Study：datacenter
  - 有个software stack的图可以看下


## Consistency（lec2）

- 定义：consistency model是一组规则定义了updates的顺序和可观测性，有tradeoff。
  - 单个对象的consistency 叫coherence，比如cpu本地内存中对象的一致性

- 没有正确与否，只有编程容易度和性能之间的tradeoff
- 分布式很难，因为要考虑数据备份，并行，容错
  - 举个互斥锁的例子，单机很简单，但是分布式里面就可能会有网络导致的可观测的问题


- Strict Consistency
  - 定义：所有操作符合全局时钟的序关系
    - 所有读能读到最新的值
    - 所有单CPU上的操作都有一个执行的时间戳
  - 可以保证critical section，有个小证明：假设法，然后违背最新读取


- Sequential Consistency

  - 背景：强一致性基本不可能实现，因为没有全局时钟


  - 定义：最接近Strict Consistency，保证操作的total order关系

    - 每个CPU里面的操作有序关系
    - 所有CPU看到的结果都要满足一个全局的序关系
  - 也可以保证critical section，因为不存在一个全局序关系
  - 需要考虑同一个对象的至少有一个是写的关系
  - 两个规则：
    - 所有CPU发请求不会reorder
    - memory是FIFO来处理数据的

  - DSM违反了上述两个规则

- Case Study：Ivy: A Shared Virtual Memory System for Parallel Computing

  - Ivy是一个共享内存系统，保证了sequential consistency
  - 实现
    - 每个节点本地负责维护一部分的page（利用cache提高性能）
    - 最新写数据的节点会成为page的owner
    - 需要一个global manager
      - 维护了所有page，copy_set，owner信息
      - owner信息维护，保证一个page只有一个owner，保证page的validation
    - 如果某个节点找不到page，会问manager谁是owner，然后直接问owner要
    - 需要写的时候，manager先invalidate所有copy，然后转移所有权
  - 可以保证critical section
  - Invariants
    - 每个page有且只有一个owner
    - owner肯定有page的一个copy
    - 如果被ower改了，别人没有copy
    - 如果被ower读了，和其它copy一致
    - manger知道所有的copy

- Release Consistency

  - 在SC里面，肯定需要一个global manager，而且所有读写操作都需要同步
    - 但是如果node访问同一个页中的不同对象，其实没必要同步，导致了false sharing的问题

  - 引入了新的变量：synchronization variables (locks)
    - 可以被acquire和release
  - 实现对了的情况，release consistency == sequential consistency
    - 获取变量先加锁
    - 放锁前所有的读写操作都结束
    - acquire和release不会reorder

  - 同步的Protocol
    - 通信内容
      - Update：改了后直接告诉别人修改后的值
      - Invalidate：改了后告诉别人改了，但是需要别人用到的时候过来读值
      - 对比：invalidate一次通信开销更小，但是可能会有miss带来的巨大开销
    - 通信时间
      - Eager：当release的时候把所有修改发出去
      - Lazy：当别人需要的时候才把修改发出去
      - 对比：Eager在acquire的时候不需要同步，Lazy可以避免不必要的同步
  - 如何实现在多个写者的情况下找到所有的modification：twinning
    - 第一次修改前创建一个page的twin
    - 后续用diff找到所有修改

- Eventual Consistency（lec3）

  - 背景：
    - SC从read/write角度维护order，RC从release的角度维护order
    - SC/RC有不足
      - 慢，操作前多需要去查看数据或者锁情况
      - 对可用性要求高，需要很多的通信
      - 某些场景不合适，如离线终端，可用性要求高，地理分布远的应用
    
  - 定义：相比SC更加乐观，认为冲突少，后续再处理成一个一致的order
  
  - 好处：性能更高
    - 写可以在被序列化之前接受
    - 读不会被阻塞（但是写可能会读出来一个旧的值）


  - 问题：Anomalies
    - Write和Write有冲突，但最终会一致
    - 会读到旧的数据，保证一定的causality


  - 实现：
    - 多台机子，每台都可以读写，定期同步
    - 如何处理写冲突？
      - 使用update log：记录所有update操作（一定要是deterministic的操作），同步操作序列Log
    
    - 如何决定顺序？
      - Update记录ID < time T, node ID> ,可以判断序关系
    
    - 如何解决不同机子的不一致情况
      - 同步时，通过rollback和replay解决
    
    - 时钟不一致，导致可能出现先删后加的情况
      - 使用逻辑时钟，如果一个机子观察到了更大的时钟id，更新本地的为最大的
    
    - Tentative问题，即何时是Eventual
      - 去中心化策略：如果看到了所有server都发了比commit point N更大的TS，说明N之前的都被commit了
        - 如果有node下线了，可能这个commit point N就不会前进。
      - 中心化策略：有一个primary服务器来确定全局的Commit order number，<CSN, local-TS, SrvID>，有CSN的write就是stable的，放在没有CSN的log前面
        - CSN一般来说是匹配TS的，但是如果同步和到达顺序不一致就有问题，以CSN为准
    
  - case study： Bayou和COPS
    
    - Bayou
      - 提出了上面说的 update functions， ordered update log， conflict solution
      - 利用了 eventual consistency， logical clock
    
    
    - COPS： Scalable Causal Consistency for Wide-Area Storage
      - 背景：现有基于Log的系统不好
        - Log的同步方式不能捕获causality
        - 可扩展性不高，所有服务器上面都存了所有数据，同步开销太大了
        - 跨node没有causality
      - Causal Consistency：偏序关系
        - 单线程内部保证序关系
        - 如果有读取关系，也有序关系
        - 如果op1 -> op2, op2 -> op3，那么 op1 -> op3
      - 设计
        - 本地提供服务，后台同步数据
        - 使用Causal + consistency（Causal+ = Causal + Conflict Handling）
        - 不同节点负责部分key
        - 使用metadata来维护causality
        - 显式跟踪dependency关系，等所有dependency都满足后再向外暴露数据
        - 使用用户库来帮用户处理请求
        - get, put_after( put + ordering metadata)
      - 可扩展性好，满足ALPS，同时提供了causal+的consistency
    


## Crash Consistency & Logging （lec4）

- 前面讲的都是concurency的问题，现在考虑failure下的一致性，即atomicity（all-or-nothing）
- stawman：使用shadow page解决？
  - 多个transaction同时编辑相同文件会出现问题

- Logging

  - 所有更新操作都记录log
  - do会产生WAL log，redo和undo log可以恢复状态

  - case study: System R

    - 所有transaction共享同一个log，append-only，同一个transaction的log用向前的链表连起来
    - Log entry里面有Transaction ID， 操作ID，同transcation的前一个操作指针，操作内容
    - Logging rules
      - WAL log：持久化操作前都要记log
      - Commit：commit的时候在log后面加个commit

    - Recovery rules
      - 从后先前，把所有没有commit的transaction加入abort log
      - 标出commit的transaction的log，从前向后redo，从后向 前undo

    - checkpoint
      - 避免太长的transaction容易被abort，缩短恢复时间
      - 实现：
        - 等到目前没有action，在log里面加入ckpt record
          - 包含所有在执行的transaction和他们最近的log的地址
        - 刷所有cache到磁盘
        - **原子**更新checkpoint root到新的ckpt record
        - 从ckpt开始恢复
    - 为什么要同时有undo和redo log
      - transaction可能很长，也又可能被abort，所以要记录undo log

    - Redo-only logs
      - 适合短的transaction，能更快提交，缺点是transaction可能要做两遍，恢复时间长，log长
      - 要求：
        - 刷状态到磁盘前要写redo log
        - 没有被commit的transaction不能刷状态

    - Undo-only logs
      - 恢复时，不用把transaction做两遍，更快，没有写放大，log空间更好，缺点是提交得等刷磁盘
      - 要求：
        - 刷状态到磁盘前要写undo log
        - commit前必须要刷状态到磁盘
    - Checkpoint避免log过长

  - case study： FSD 
    - 文件系统在更新数据时需要考虑crash consistency
    - 同步的元数据更新有问题：
      - 太慢了
      - 需要检查整个磁盘
      - 有些操作没有commit
    - Logging file system：redo-only with checkpoint
      - write disk cache in memory
      2. Append log records

      3. Append commit log record

      4. Wait for all its log records to reach the disk（background）

      5. Write disk cache to disk（background）

      6. Update checkpoint（background）


## Concurrency Control （lec5）

- 指多个transaction并行执行的ACID保证

- Serializability

  - 符合全局的事务执行顺序

  - 实现：
    
    - strawman
      - 全局锁粒度大，太慢了
      - 操作前后加锁不能保证Serializability，会出现uncommitted read
      - 读马上放锁，写拿着到commit，有可能出现non-repeatable read
    
    - 2PL
      - 两阶段：growing（整个transaction）拿锁，shrinking（commit）放锁
      - 优化，使用读写锁
      - 可能有死锁
        - 顺序拿锁（不是很实际）、死锁检查
    
      - 可能有幻读（phantom）
        - 要保证新插入的不可见：用predicate lock或者range lock（实践中为了性能允许出现）
    
      - 问题：
        - 性能不行，只适合长时间只读的事务
        - 需要分布式的死锁检测
        - 读可能会阻塞 update
    
    
    - MVCC
      - 无锁设计，每个数据有多个版本
      - 先buffer 写，然后commit的时候看是否能写入，避免写写冲突（检测方式就是看数据的版本）
      - Snapshot isolation
        - 没有 read uncommitted 和 unreaptable-read的问题
        - 不等于serializability，有情况符合SI但不符合serializability
          - T1 读 X 写 Y；T2 读Y 写 X
          - R(X) = X0, W(Y) = Y1; R(Y) = Y0, W(X) = X1


## 2PC（lec6）

- 解决了在分布式系统下如何保证atomicity和concurrency control的问题。
- 需要一个Transaction Coordinator (TC)
- 两个phase
  - Voting：喊大家prepare
  - Committing：所有人提交

- timeout怎么办
  - Termination Protocol优化
    - 只有在timeout后使用
    - participants等不到TC的话直接去问其它node，从其它人那边获取信息

- Reboot怎么办？
  - 看ppt

- case study：SINFONIA: A New Paradigm for Building Scalable DS
  - 问题：旧的系统的transaction太昂贵了
  - 目标：方便系统程序员编程
  - 架构是用户库和memory node，使用transaction
  - mini-transaction
    - network更少，牺牲了flexible，有一定要求
    - 操作：
      - 用cmp检查数据
      - 如果检查数据通过，用read那数据，用write写数据

    - 可以用来做院原子操作或者锁操作等

  - 2PC protocol修改
    - 把TC放到application里面，节省了RTT
    - 问题：TC很容易crash
    - 解决：TC不维护log，用participant来维护数据，通过memory node 投票来决定是否commit或者问memery node 是否commit来一个transaction
      - 问题是memory node 必须要靠谱




## Distributed Consensus（lec7）

- 分布式容错怎么做，简单方法就是replication
- Replicated State Machine （RSM）
  - 初始状态相同，通过相同的操作，操作一定是确定性的，所有server都会到达相同的状态
  - 可以在backup读不能保证sequential consistency
    - Primary和Backup可以保证
      - 能从backup读吗
        - 可以，需要一些方法来保证一致性，如2PC
  - primary挂掉后，谁成为新的primary，需要consensus

- Paxos
  - 没有中心化的coordinator，考虑所有网络错误，在majority alive情况下work
  -  Client, Proposer, Acceptor, Learner
    - Phase 1：Prepare
      - Leader： propose, N比proposer见到的最大proposal大
      - Acceptor：如果N比见到的都大，回复见到最大的proposal和值，并提供promise；否则，忽略

    - Phase 2：Accept
      - Leader：收到majority，设置V，发送accept request给Leader和Learner；否则，重新来
      - Acceptor：如果promise有效，设置V，返回accept；否则，忽略

    - Phase 3： Learn
      - Leader：发送decide给所有人
      - Learner：回复用户或者执行操作

  - 问题：可能不会terminate




## Distributed Storage（lec8）

- Network File System
  - 模式
    - upload/download
      - 好处：简单
      - 坏处：浪费带宽，可能导致空间问题，consistency有问题
    - create, delete, read, write, etc …
      - 好处：细粒度，一致性有保障
      - 坏处：server网络congestion，因为server服务时间长，重复请求多次相同数据
  - 通过VFS实现
  - Session semantics：所有人能看到现有打开的版本（存在cache中），最后一个更新者win
  - State in server
    - Stateful: request更短，性能更好，可以做cache和lock
    - Stateless：crash更好处理，不需要open/cloase，不能做locking，delete难做
  - Cache
    - Write-through： use invalidation
    - Delayed writes: periodically update remote
  - NFS
    - Mount Protocol: client发lookup请求，server返回handle，后续请求（有16个函数）都带上handle
    - 比本地慢
      - client会cache文件数据，文件属性，路径
      - server也有cache，使用write-through 方法 （直接写磁盘）
    - 使用timestamp来处理inconsistency，获取时或者定期更新，在关闭时写回
    - 优化：使用更大的块来传输数据，提前读可能会被访问的数据
    - 问题：一致性，依赖一致的时间，append不一定能用，lock没用，没有reference count
    - 还有优化看ppt p25
  - case study：GFS
    - 假设：
      - 大多数都是append，顺序读比较多
      - 读写都很大
    - API：create/delete/open/close/read/write + snapshot/append
    - 架构：
      - 一个master+ 多个chunkserver
      - 文件由chunk（64MB）组成，有备份（3）和checksum
      - master管理metadata等信息，负责chunk的管理，也有replication，心跳机制
    - 操作
      - 读：client和master交互获取metadata，然后直接问chunkserver要数据（避免master的congestion）
      - 写：master选择一个chuckserver成为primary，给chunk lease，增加chunk version并通知其它replicas，具体写入时有两个phase
        - send data: 发送数据，client给最近的replica发数据，然后遍历chunkserver，但目前只把数据放在内存
        - write data：commit，当所有的副本都返回已经接收数据成功后，client会向primary发送一个写请求。primary会为每一个数据更改的请求附加一个序列号，数据更改是按照序列号的顺序执行的。primary将数据更改同步到其他副本中，副本也是按照序列号执行数据更改操作。primary接收到其他副本回复的数据操作完成，返回client结果。期间发生的所有错误都会报给client。
        - 数据流和控制流不同，version用来检查stale data
    - 所有文件数据都没有cache，client有metadata的cache
    - master
      - 在内存中存储所有metadata
      - namespace，name-to-chunk的映射（这部分会有在log里面）
      - 不持久化chunk在哪台服务器上，直接去问就行，为了避免一致性问题
    - Large chunk：减少通讯，保持TCP连接，减少master中存储的metadata数目
  - case study： Hadoop DFS
    - 架构：master+ slave
    - 1 namenode：负责namespace和权限管理
    - n datanode：存数据（64MB一个block）
    - 其实和上面差不多

## Distributed Programming（lec9）

- MapReduce
  - Master + worker
  - Map + Reduce
  - 实现：
    - split file into chunks
    - fork processes
    - map task
    - create intermediate files （ in map server）
    - partitioning（for reduce servers）
    - sorting intermediate data（reduce server 做，为了给reduce function做输入）
    - reduce task
    - return to user
  - Locality
    - 尽量使map worker是对应的chunk server（GFS场景下）
  - Fault Tolerance
    - worker心跳，死了就re- execute
    - master和GFS一样
  - Straggler，最慢的机子怎么处理
    - 多个机子一起做一样的活，谁先好用谁的
  - Intermediate Key/Value Pairs
    - mapper：节省网络，map失败不会产生部分数据，reduce失败可以再拿
    - reduce：可以把传输时间给overlap了

- Dryad
  - 背景：MapReduce处理网页数据很慢，无法提供实时分析，没有连续性
  - 用DAG图来表示计算，点是计算（job），边是通讯
  - Structure：
    - Job Manager：任务调度
    - name server：发现可以用的服务器
    - Daemon process：处理job

  - 优点：
    - 大任务友好，本地化更好
    - 操作更加flexible

  - 缺点：
    - 比MapReduce复杂


## Graph-parallel Computation（lec10）

- 背景：
  - 大数据而且数据之间有关系，如PageRank（等于所有指向这个点的点所有出边的权重除以出边数目之合）
  - MapReduce无法处理有关系的数据，而且没有对iteration优化

- case study：Pregel
  - Bulk Synchronous Parallel （BSP）模式：自己算，算完同步，每次同步算一个step
  - Think like a vertex
  - 架构：
    - master：负责统计数据和barrier synchronization
    - 节点：干活，互相通讯，收到数据active，自己判断deactive

- case study：GraphLab
  - BSP有问题：因为是同步的，所以计算次数很多，而且最慢的机子会拉低整体效率
  - 数据以图的格式表示
    - 单机上用array表示
    - 分布式场景中，将图partition到不同机子上
      - Ghost vertices保存了图的联系，使用HPC Graph partitioning tools (比如ParMetis)

  - Update function：用户指定，对顶点做操作
  - scheduler：负责决定计算的顺序
    - Round Robin, FIFO, Priority

  - Sequential Consistency (越前面越严格，性能越差)
    - Full consistency > Edge Consistency > Vertex Consistency
    - 通过两个引擎实现：染色引擎和分布式锁引擎


## DrTM（lec11）

- 背景：分布式transaction很重要，但现有方案要不性能差，要不可扩展性差
  - 原因是很多时间被一些mechanism占用了
- 硬件feature
  - HTM（Hardware Transactional Memory）在硬件角度实现了transaction
    - Restricted Transactional Memory (RTM)
    - 强原子性
  - RDMA
    - one-sided verb有强一致性
- HTM有限制：只在单机使用，如何变成分布式
  - 利用2PL，分为local tx和dist tx，local的优先级高
- RDMA语义有限制：
  - 使用lease-based protocol：其实就是租约锁
- TODO，再仔细看看



### Distributed Graph Partitioning（lec12）

- case study： PowerLyra	

  - 问题背景：locality和parallelism有问题，因为power law，存在大节点（skewed graph），目前方案没有考虑这种情况

  - Hybrid 计算和图划分

    - 图划分（Hybrid- cut）

      - 提供一个threshold，标志low degree和high degree
      - 对low-degree节点，hash点到不同partition上面
      - 对high-degree节点，hash边到不同的partition上面

    - 图计算

      - 对于high-degree节点，用GAS（gather-apply-scatter）模型，强调parallelism
      - 对于low-degree节点，用locality，  主要是gather阶段不一样

      



### Disk-based Parallel Processing（lec13）

- case study： GraphChi

  - 目的：在单机上利用磁盘实现大数据图计算

  - 主要挑战：磁盘访问非常随机

  - 方案：Parallel Sliding Windows (PSW)

    - 在只需要很少量的随机访问的情况下，PSW就可以更新边的值。基本的操作分为3部分：1 从磁盘中加载子图；2. 更新顶点和边的操作；3. 将更新之后的数据写到磁盘上面；

      - 加载子图；一个图G会被分割为不相交的区间，每一个区间会有一个对应的shard的结构，这里面保存的是以这个区间内的顶点为目的顶点的边的集合，并且会根据它们的源顶点来排序。分割图的时候，要考虑到尽量地将它们分割均匀，而且要都能加载到内存里面，不能太大。这里分割考虑的大小主要是根据边，要保证边的数量尽量一致，而对于顶点的数量，则可以不同。GraphChi执行的时候，是以internval为单位执行的。对与一个interval顶点的出边，它们会位于其它的shards结构的连续的块内。这里就体现了根据它们的源顶点来排序的意义了。

        ```pseudocode
        foreach iteration do
          shards[] ← InitializeShards(P) 
          for interval ← 1 to P do
             /* Load subgraph for interval, using Alg. 3. 
             * Note, that the edge values are stored as pointers to the loaded file blocks. 
             */
            subgraph ← LoadSubgraph (interval) 
            parallel foreach vertex ∈ subgraph.vertex do
              /* Execute user-defined update function, which can modify the values of the edges */
              UDF updateVertex (vertex) /* Update memory-shard to disk */
              shards[interval].UpdateFully()
              /* Update sliding windows on disk */
            end
            s ∈ 1,..,P, s != interval do 
              shards[s].UpdateLastWindowToDisk()
            end 
          end
        end
        ```

      - 并行更新，如上面的伪代码所示，内存在的更新操作是并行进行的。这里的更新的函数一般为用户定义。由于是并行进行的，就要处理一些情况下可能出现的并发的问题
      - 将更新之后的数据写入磁盘，将更新完整之后，需要将更新之后的数据写入磁盘，对于非这个interval的shards，只需要写入更新的一部分即可。这样每个interval的一次操作需要P(分区的数量)的非顺序写入操作

- 设传输的块大小为B，边的数量为|E|，IO的复杂度与这两个的除直接相关。另外最坏的情况下，还有$P^2$次的非顺序的磁盘操作，这样就有：$2|E|/B ≤ Q_B E ≤ 4|E|/B+ Θ(P^2)$

- Evolving graph

  - Shard里面的边不要求是严格有序的，它讲Shard分为P个部分，每个部分分配一个内存中的edge-buffer(p, j)结构。向里面添加边的时候，当超过一定数量的时候，就会分裂为2个新的部分。
  - 对于删除边的操作，是在运行的时候标记为删除，在写入到磁盘的时候去真正的删除。


### Tiled- MapReduce（lec14）

- 背景：mapreduce在multicore场景下不好
  - high memory usage
  - poor data locality
  - strict dependency barriers

- Tiled- MapReduce
  - 把大的mapreduce job分成小的，每次就做一次小的
    - 要求：reduce任务要求符合交换律和结合律
    - 不断地map然后combine（就是一个小的reduce），最后交给reduce

- 优化
  - Input Data Buffer Reuse 
    - 在combine里面复制需要的数据，只保存现有sub-job需要的数据，input buffer在所有sub-job可以复用

  - NUCA/NUMA-aware Scheduler 
    - 消除远程调用，所有sub-job在同一个chip

  - Software Pipeline
    - Overlap the Combine phase of the current sub-job and the Map phase of its successor


### Graph-aware Fault Tolerance（lec15）

- 相关工作
  - mapreduce直接re-execute ，图计算直接有依赖，必须记录所有过程
  - spark记录操作逻辑和数据依赖关系，然后re-execute ，太粗粒度了
  - Trinity, PowerGraph, Pregel, etc.使用checkpoint，记录所有状态，有问题：
    - 太多状态，同步压力大
    - recovery太慢
  - 思想：用备份来实现fault tolerance，可以实现小的overhead和快的recovery
  - 每个节点有f个备份（mirror）
    - 优化：selfish vertices：不存数据，通过邻居直接算出来
  - recovery：
    - Master recovers replicas
    - If the master crashed, the mirror recovers the master and replicas
  - Rebirth
    - Migrate tasks to surviving machines
      - Mirrors upgrade to masters and broadcast
      - Reload missing graph structure and reconstruct





