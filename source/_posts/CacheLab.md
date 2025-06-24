---
title: CacheLab
date: 2024-09-10
categories:
- CMU15-213
index_img: /Pictures/CMU15-213/CacheLab.jpg
banner_img: /Pictures/CMU15-213/CacheLab.jpg
---



# CacheLab

## Lab概述：

在本实验中，学生将处理两个名为 csim.c 和 trans.c 的 C 文件。实验分为两部分： (a) 部分涉及在 csim.c 中实现缓存模拟器。 (b) 部分涉及编写一个函数，该函数计算给定矩阵的转置，目标是减少模拟缓存中的未命中次数。

## 相关工具：

使用autograders需要执行以下指令，首先需要编译：

```bash
make clean
make
```

检查缓存模拟器的正确性：

```bash
./test-csim
```

检查你的转置函数的正确性和性能：

```bash
./test-trans -M 32 -N 32
./test-trans -M 64 -N 64
./test-trans -M 61 -N 67
```



## Part A

在这一部分中，我们需要在`csim.c`文件中编写一个缓存模拟器，实现对不同参数$(S,E,B,m)$缓存的模拟，并统计缓存命中/不命中/替换的数量。

用来作为输入的文件保存在`./traces`目录下，这些文件具有一定的格式：

```tex
I 0400d7d4,8
 M 0421c7f0,4
 L 04f6b868,8
 S 7ff0005c8,8
```

每行表示一次或两次内存访问。字母表示操作类型：I-指令加载，M-数据修改（例如一次数据加载后跟随一次数据存储），L-数据加载，S-数据存储。后面为64位地址和访问的字节数。

Lab为我们提供了一个参考二进制文件`./csim-ref`。该文件的使用遵循下面的格式：

```bash
./csim-ref [-hv] -s <s> -E <E> -b <b> -t <tracefile>
```

**-t** 指定一个trace file，然后在分析里面每个操作对缓存的影响，是hit还是miss，有没有eviction
**-h** 表示打印帮助信息，**-v**表示打印每个操作的分析结果（可选）
**-s** 指定**组索引的位数**，**-E** 指定**每组的行数**，**-b** 指定**块偏移的位数**

例如，我们执行：`./csim-ref -v -s 4 -E 1 -b 4 -t traces/yi.trace`，会得到以下结果：

```bash
L 10,1 miss
M 20,1 miss hit
L 22,1 hit
S 18,1 hit
L 110,1 miss eviction
L 210,1 miss eviction
M 12,1 miss eviction hit
hits:4 misses:5 evictions:3
```

我们要做的就是依葫芦画瓢，自己将`csim-ref`实现即可。



### 读取命令行参数

首先我们的程序需要对不同的命令行参数进行处理，所以我们先构建出大致的框架，之后再完善各个模块的功能。查询相关资料，解析命令行参数可以手动对`argc`和`argv[]`进行判断，但更便捷的方法是使用`<unistd.h>`标准库中的`getopt()`函数。

`int getopt(int argc, char * const argv[], const char *optstring);`

**`argc`**: 命令行参数的数量，通常是 `main` 函数中的 `argc`。

**`argv`**: 命令行参数的数组，通常是 `main` 函数中的 `argv`。

**`optstring`**: 包含有效选项的字符列表。每个选项后面可以跟一个冒号 (`:`) 表示该选项需要一个参数。多个选项用空格分隔。

**返回值**：返回下一个命令行选项字符，如果遇到无效选项或未提供必要的参数，返回 `?`。当没有更多选项时，返回 `-1`。

我们得到：

```c
#include "cachelab.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void printhelp()
{
    char helper[1024] = "Usage: ./csim-ref [-hv] -s <num> -E <num> -b <num> -t <file>\n"
                        "Options:\n"
                        "-h         Print this help message.\n"
                        "-v         Optional verbose flag.\n"
                        "-s <num>   Number of set index bits.\n"
                        "-E <num>   Number of lines per set.\n"
                        "-b <num>   Number of block offset bits.\n"
                        "-t <file>  Trace file.\n\n"
                        "Examples:\n"
                        "linux>  ./csim-ref -s 4 -E 1 -b 4 -t traces/yi.trace\n"
                        "linux>  ./csim-ref -v -s 8 -E 2 -b 4 -t traces/yi.trace\n";
    printf("%s",helper);
}

int verbose = 0;
int s,E,b = 0;
char* tracefile;
FILE *fp = NULL;

int main(int argc,char* argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "hvs:E:b:t:")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printhelp();
                return 0;
            case 'v':
                verbose = 1;
                break;
            case 's':
                s = atoi(optarg);
                break;
            case 'E':
                E = atoi(optarg);
                break;
            case 'b':
                b = atoi(optarg);
                break;
            case 't':
                sscanf(optarg, "%s", tracefile); 
                break;
        }
    }

    // check necessary parameters
    if(!s || !E || !b)
    {
        printf("./csim-ref: Missing required command line argument\n");
        printhelp();
        return 0;
    }

    // check whether file can be opened successfully
    fp = fopen(tracefile, "r");
    if(!fp)
    {
        printf("%s: No such file or directory\n", tracefile);
        return 0;
    }

    printSummary(0, 0, 0);
    return 0;
}
```

包含了命令行参数的读取和特殊情况的检查，运行检测能够正常运行。



### 定义缓存

接下来我们需要定义缓存。根据缓存的定义，由$S=2^s$组，每组$E$行，每行$B=2^b$块。由于我们不需要处理缓存保存的实际内容，因此不需要定义块是什么。

对于每一行，由三部分组成：一个有效位（0/1），一个标签，B个数据块（此处无需在意）。此外需要注意的是，由于缓存的替换策略为**最近最少使用(Least-recently used, LRU)**，也就是说，会替换**最后一次访问时间最久远**的那一行。因此需要在定义行时，最好增加一个**时间戳stamp**表示最后一次访问该行的时间，帮助我们确定哪行需要被替换。

```c
typedef struct
{
    int valid;
    unsigned int tag;
    time_t stamp;
}line,*line_ptr;
```

获取时间戳，我们需要引入系统库`<time.h>`并编写以下函数：

```c
time_t get_current_time()
{
    time_t current_time;
    time(&current_time);
    return current_time;
}
```

接着定义cache并初始化，注意到这里的cache可以抽象为组（set）的数组，而组又是line的数组，因此cache是一个对于行的二维数组：

```c
// define cache and initialize
line_ptr* cache;

cache = (line_ptr*) malloc(sizeof(line_ptr)*(1 << s));
for(int i = 0; i < (1 << s); i++)
{
    cache[i] = (line_ptr) malloc(sizeof(line) * E);
    for(int j = 0; j < E; j++)
    {
        cache[i][j].valid = 0;
        cache[i][j].tag = 0;
        cache[i][j].time_stamp = 0;
    }
}
```



### 读取并处理操作

由于我们无需处理指令，所以可以直接跳过对I指令的读取。对于M，S，L指令，我们用格式匹配分隔出指令类型、地址和大小，由于M包含两部操作，我们将其分隔开特殊处理：

```c
// read operations
    int size;
    char operation;
    size_t address;

    while (fscanf(trace_file, "%s %lx,%d\n", &operation, &address, &size) == 3) 
    {
        if (v) {
            printf("%c %lx,%d ", operation, address, size);
        }
        switch (operation) {
        case 'I':
            continue;
        case 'M': 
            useCache(address, 1);
            break;
        case 'L': 
        case 'S': 
            useCache(address, 0);
        }
    }
```

接着处理指令，主要是将地址进行分割：0~b-1位为块偏移，b-1~b+s-1位为组索引，最后t=m-s-b位为标记，我们可以用位运算进行分离：1）首先去除b位块偏移，即`rest = address >> b`，2）使用mask（低s位全为1）保留前s位：`mask = (1 << s) - 1`，3）`set_index = rest & mask`。

同理，我们进行位运算分离标记：`tag = address >> (b+s)`。

分隔后转为十进制，所以我们直接用`int`类型保存即可。

```c
void useCache(size_t address, int is_modifyed)
{
    int set_index = address >> b & ((1 << s) - 1);
    int line_index;
    int tag = address >> (b+s);
    bool isHit = false;

    line_ptr current_set = cache[set_index];

    // find line
    for(int i = 0; i < E; i++)
    {
        if(tag == current_set[i].tag && current_set[i].valid)
        {
            isHit = true;
            line_index = i;
            break;
        }
    }

    // hit
    if(isHit)
    {
        hit++;
        // if this operation is Modify, there is one more Write operation.
        hit+= is_modifyed;
        if(verbose)
        {
            printf("hit ");
        }
        cache[set_index][line_index].time_stamp = get_current_time();
    }
    // miss
    else
    {
        miss++;
        if(verbose)
        {
            printf("miss ");
        }
        // find LRU line
        line_index = 0;
        for(int i = 0; i < E; i ++)
        {
            if(cache[set_index][i].time_stamp < cache[set_index][line_index].time_stamp)
            {
                line_index = i;
            }
        }
        if(cache[set_index][line_index].valid)
        {
            ++eviction;
            if(verbose)
            {
                printf("eviction ");
            }
        }
        cache[set_index][line_index].valid = 1;
        cache[set_index][line_index].tag = tag;
        cache[set_index][line_index].time_stamp = get_current_time();
    }
    if(verbose)
    {
        putchar('\n');
    }
}
```

整体逻辑还是很简单的，主要是判断hit和miss，对于M操作，由于hit之后的write操作是一定hit的，所以在初始传入条件的时候用变量保存信息就行。

下面给出完整代码：

```c
#include "cachelab.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

void printhelp()
{
    char helper[1024] = "Usage: ./csim-ref [-hv] -s <num> -E <num> -b <num> -t <file>\n"
                        "Options:\n"
                        "-h         Print this help message.\n"
                        "-v         Optional verbose flag.\n"
                        "-s <num>   Number of set index bits.\n"
                        "-E <num>   Number of lines per set.\n"
                        "-b <num>   Number of block offset bits.\n"
                        "-t <file>  Trace file.\n\n"
                        "Examples:\n"
                        "linux>  ./csim-ref -s 4 -E 1 -b 4 -t traces/yi.trace\n"
                        "linux>  ./csim-ref -v -s 8 -E 2 -b 4 -t traces/yi.trace\n";
    printf("%s",helper);
}

int verbose = 0;
int s,E,b = 0;
unsigned hit = 0, miss = 0, eviction = 0;
FILE *trace_file;
char* trace_filename;

// define line
typedef struct
{
    int valid;
    int tag;
    time_t time_stamp;
}line,*line_ptr;


line_ptr* cache;

// get timestamp
time_t get_current_time()
{
    time_t current_time;
    time(&current_time);
    return current_time;
}

void useCache(size_t address, int is_modifyed)
{
    int set_index = address >> b & ((1 << s) - 1);
    int line_index;
    int tag = address >> (b+s);
    bool isHit = false;

    line_ptr current_set = cache[set_index];

    // find line
    for(int i = 0; i < E; i++)
    {
        if(tag == current_set[i].tag && current_set[i].valid)
        {
            isHit = true;
            line_index = i;
            break;
        }
    }

    // hit
    if(isHit)
    {
        hit++;
        // if this operation is Modify, there is one more Write operation.
        hit+= is_modifyed;
        if(verbose)
        {
            printf("hit ");
        }
        cache[set_index][line_index].time_stamp = get_current_time();
    }
    // miss
    else
    {
        miss++;
        if(verbose)
        {
            printf("miss ");
        }
        // find LRU line
        line_index = 0;
        for(int i = 0; i < E; i ++)
        {
            if(cache[set_index][i].time_stamp < cache[set_index][line_index].time_stamp)
            {
                line_index = i;
            }
        }
        if(cache[set_index][line_index].valid)
        {
            ++eviction;
            if(verbose)
            {
                printf("eviction ");
            }
        }
        cache[set_index][line_index].valid = 1;
        cache[set_index][line_index].tag = tag;
        cache[set_index][line_index].time_stamp = get_current_time();
    }
    if(verbose)
    {
        putchar('\n');
    }
}

int main(int argc,char* argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "hvs:E:b:t:")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printhelp();
                return 0;
            case 'v':
                verbose = 1;
                break;
            case 's':
                s = atoi(optarg);
                break;
            case 'E':
                E = atoi(optarg);
                break;
            case 'b':
                b = atoi(optarg);
                break;
            case 't':
                trace_filename = optarg;
                trace_file = fopen(trace_filename, "r");
                break;
            default:
                printhelp();
                return 0;
        }
    }

    // check necessary parameters
    if(!s || !E || !b)
    {
        printf("./csim-ref: Missing required command line argument\n");
        printhelp();
        return 0;
    }

    // check whether file can be opened successfully
    if(!trace_file)
    {
        printf("%s: No such file or directory\n", trace_filename);
        return 0;
    }

    // define cache and initialize
    cache = (line_ptr*) malloc(sizeof(line_ptr)*(1 << s));
    for(int i = 0; i < (1 << s); i++)
    {
        cache[i] = (line_ptr) malloc(sizeof(line) * E);
        for(int j = 0; j < E; j++)
        {
            cache[i][j].valid = 0;
            cache[i][j].tag = 0;
            cache[i][j].time_stamp = 0;
        }
    }

    // read operations
    int size;
    char operation;
    size_t address;

    while (fscanf(trace_file, "%s %lx,%d\n", &operation, &address, &size) == 3) 
    {
        if (verbose) {
            printf("%c %lx,%d ", operation, address, size);
        }
        switch (operation) {
        case 'I':
            continue;
        case 'M': 
            useCache(address, 1);
            break;
        case 'L': 
        case 'S': 
            useCache(address, 0);
        }
    }

    free(cache);
    printSummary(hit,miss,eviction);
    return 0;
}
```

运行测试，结果如下所示时说明通过：

```shell
                        Your simulator     Reference simulator
Points (s,E,b)    Hits  Misses  Evicts    Hits  Misses  Evicts
     3 (1,1,1)       9       8       6       9       8       6  traces/yi2.trace
     2 (4,2,4)       2       5       2       4       5       2  traces/yi.trace
     3 (2,1,4)       2       3       1       2       3       1  traces/dave.trace
     2 (2,1,3)     163      71      67     167      71      67  traces/trans.trace
     0 (2,2,3)     192      42      34     201      37      29  traces/trans.trace
     0 (2,4,3)     206      30      14     212      26      10  traces/trans.trace
     3 (5,1,5)     231       7       0     231       7       0  traces/trans.trace
     4 (5,1,5)  264181   21775   21743  265189   21775   21743  traces/long.trace
    17

TEST_CSIM_RESULTS=17
```



## Part B

在`trans.c`文件中，为我们提供了一个示例矩阵转置函数：

```c
char trans_desc[] = "Simple row-wise scan transpose";
void trans(int M, int N, int A[N][M], int B[M][N])
{
    int i, j, tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            tmp = A[i][j];
            B[j][i] = tmp;
        }
    }    

}
```

这个函数很简洁，但是性能并不令人满意，因为这种模式会造成大量的缓存不命中。我们要做的便是编写一个高性能的转置函数，减少缓存不命中数量。

实验在$(s,E,b)=(5,1,5)$的条件下进行，分为$32\times32,\ 64\times64,\ 61\times67$三种大小的矩阵。

又到了我们喜闻乐见的优化题。在书中对这个问题进行了简单的解释：由于矩阵在内存中的保存形式为行优先存储的一维数组，对于A矩阵和B矩阵而言，两者一个是行优先，一个是列优先，导致存储的块冲突造成缓存不命中。

例如，假设我们以第一问$32\times32,\ (s,E,b)=(5,1,5)$为例，每个块存放8个整数，假设我们要将$A[0][1]\rightarrow B[1][0]$，而$A[0][1]$存放在高速缓存的行0，$B[1][0]$存放在行4。然后是$A[0][2]\rightarrow B[2][0]$，根据我们的计算方法，$A[0][2]$还是保存在行0，缓存命中，而$B[2][0]$保存在行8，造成不命中。而当我们写到$B[1][1]$时，以高速缓存的容量原来的行4早已被驱逐，所以对于B的每次写都是不命中的，这就有1024次。而对于A我们每读8次会有一次不命中，一共就是128次。

在这里还要考虑主对角线上的情况，也就是$A[i][i]\rightarrow B[i][i]$的情况，由于此时两者必然占用同一行编号，因此在  **写B----读A的下一个元素**  这个过程中，由于含$B[i][i]$的块驱逐原有含$A[i][i]$的块，这样在读取$A[i][i+1]$时会有一次不命中，所以一共是1024+128+32=1184次不命中。

在书中的旁注提到了分块（Blocking）技术，就是将数据分割为大的片（chunk），称为块（这里的块与高速缓存中的块不同）。这样能够让一个片加载到高速缓存中，进行所有的读和写操作后丢掉这个片，再读取下一个片。我们这里主要利用这种方法进行优化。

### $32\times32$

根据实验的环境，块的大小为$2^5=32$ bytes ，也就是一行能保存8个int类型的数据。而每组1行，一共32组，也就是一共能够保存32行256个int。

在这里，我们最好是首先去查找A和B存储的地址，来判断一下高速缓存的每一行应该存放哪些元素。在这里A存储在0x30a0c0位置，B存储在0x34a0c0位置，推算得出$A[0][0]\sim A[0][7]$保存在同一行，并且$A[i][j]$与$B[i][j]$对应的组索引是一样的。

如果我们按$8\times8$进行分块操作的话，矩阵每一行8个元素正好能保存到高速缓存的一行中，并且高速缓存的容量足以保存$8\times 8$的所有元素：

```c
void trans1(int M, int N, int A[N][M], int B[M][N]){
    int i, j;
    int ii, jj;
    int size = 8;
    int maxI = N / size, maxJ = M / size;
    for(i = 0; i < maxI; ++i){
        for(j = 0; j < maxJ; ++j){
            int minII = i * size, maxII = (i + 1) * size;
            int minJJ = j * size, maxJJ = (j + 1) * size;
            for(ii = minII; ii < N && ii < maxII; ++ii){
                for(jj = minJJ; jj < M && jj < maxJJ; ++jj){
                    B[jj][ii] = A[ii][jj];
                }
            }
        }
    }
}
```

结果：`misses:344`，理论最小值为256，这说明还有优化空间，而优化的关键就在于我们之前分析的对角线元素的冲突。所以我们要做的就是想办法让写B的对角元素再读取A下一个元素时不造成不命中，一个朴素的想法就是我们在读取每一行时，将这一行元素保存下来（实际是存在寄存器中）。

```c
void trans_1(int M, int N, int A[N][M], int B[M][N])
{
	int ii, jj, i, val1, val2, val3, val4, val5, val6, val7, val0;
	for(jj = 0; jj < 32; jj += 8)
		for(ii = 0; ii < 32; ii += 8)
		{
			for(i = ii; i < ii + 8; i++)
			{
				a0 = A[i][jj];
				a1 = A[i][jj + 1];
				a2 = A[i][jj + 2];
				a3 = A[i][jj + 3];
				a4 = A[i][jj + 4];
				a5 = A[i][jj + 5];
				a6 = A[i][jj + 6];
				a7 = A[i][jj + 7];
				B[jj][i] = a0;
				B[jj + 1][i] = a1;
				B[jj + 2][i] = a2;
				B[jj + 3][i] = a3;
				B[jj + 4][i] = a4;
				B[jj + 5][i] = a5;
				B[jj + 6][i] = a6;
				B[jj + 7][i] = a7;
			}
		}
}
```

最后是287次miss，达到满分要求。

### $64\times 64$

我们依旧是查看地址：A数组首地址为0x30b120 ，B数组首地址为0x34b120 。推算得到$A[i][j]$与$A[i+4][j]$就发生冲突了。

其实从矩阵的大小我们也可以看出一些端倪：由于我们的高速缓存最多存256个数据，对于$32\times32$的矩阵来说就是最多存8行，即第0~7行与第8~16行组索引一定相同。这时我们进行$8\times8$的分块可以保证不发生冲突。而对于$64\times64$的矩阵，组索引每4行循环一次，所以如果我们头铁依旧使用$8\times8$分块，最好也就是$4\times8$的效果。

所以我们直接采用$4\times4$分块：

```c
void trans_2(int M, int N, int A[N][M], int B[M][N])
{
	int a0, a1, a2, a3;
	for(int j = 0; j < M; j += 4)
	{
		for(int i = 0; i < N; i += 4)
		{
			for(int ii = i; ii < i + 4; ii++)
			{
				a0 = A[ii][j];
				a1 = A[ii][j + 1];
				a2 = A[ii][j + 2];
				a3 = A[ii][j + 3];
				B[j][ii] = a0;
				B[j + 1][ii] = a1;
				B[j + 2][ii] = a2;
				B[j + 3][ii] = a3;
			}
		}
	}
}
```

但结果离满分还是有一定差距。

目前网络上[最优的思路](https://zhuanlan.zhihu.com/p/387662272)是进行二重分块：先进行$8\times 8$的分块，再将$8\times8$分为$4\times4$的小块。

这样做的原理是“区块借用”。这里的借用具体而言分为两种：对角线上$8\times8$的区块借用和非对角线上块内$4\times4$的区块借用。

举个例子，假设我们取一个非对角线上的$8\times8$块，内部分为4个$4\times4$小块，并且上面两个小块和下面两个小块是存在冲突的。我们每次可以读取8个数据，但是最多写入4个（因为写后4个数据时，B矩阵会将下面四行存入缓存导致冲突），还有4个数据没有利用就造成了浪费。所以我们临时“借用”一下B矩阵右上角的小块来存放这些数据，这样就避免了冲突浪费。

而对于对角线上的块，我们就要借用其局部最接近的$8\times8$块来避免冲突。

具体的思路和讲解请点击超链接，我这里提供的是仅优化非对角线块的代码：

```c
void trans2(int M, int N, int A[N][M], int B[M][N]){
    int i, j, ii, jj;
    int a0, a1, a2, a3, a4, a5, a6, a7;
    for(i = 0; i < N; i += 8){
        for(j = 0; j < M; j += 8){
            for(ii = i, jj = j; ii < i + 4; ++ii, ++jj){
                a0 = A[ii][j];
                a1 = A[ii][j + 1];
                a2 = A[ii][j + 2];
                a3 = A[ii][j + 3];
                a4 = A[ii][j + 4];
                a5 = A[ii][j + 5];
                a6 = A[ii][j + 6];
                a7 = A[ii][j + 7];
                B[jj][i] = a0;
                B[jj][i + 1] = a1;
                B[jj][i + 2] = a2;
                B[jj][i + 3] = a3;
                B[jj][i + 4] = a4;
                B[jj][i + 5] = a5;
                B[jj][i + 6] = a6;
                B[jj][i + 7] = a7;
            }
            for(ii = 0; ii < 4; ++ii){
                for(jj = ii + 1; jj < 4; ++jj){
                    a0 = B[j + ii][i + jj];
                    B[j + ii][i + jj] = B[j + jj][i + ii];
                    B[j + jj][i + ii] = a0;
                }
            }
            for(ii = 0; ii < 4; ++ii){
                for(jj = ii + 1; jj < 4; ++jj){
                    a0 = B[j + ii][i + 4 + jj];
                    B[j + ii][i + 4 + jj] = B[j + jj][i + 4 + ii];
                    B[j + jj][i + 4 + ii] = a0;
                }
            }
            for(ii = 0; ii < 4; ++ii){
                a0 = B[j + ii][i + 4];
                a1 = B[j + ii][i + 5];
                a2 = B[j + ii][i + 6];
                a3 = B[j + ii][i + 7];
                
                B[j + ii][i + 4] = A[i + 4][j + ii];
                B[j + ii][i + 5] = A[i + 5][j + ii];
                B[j + ii][i + 6] = A[i + 6][j + ii];
                B[j + ii][i + 7] = A[i + 7][j + ii];
                
                B[j + 4 + ii][i] = a0;
                B[j + 4 + ii][i + 1] = a1;
                B[j + 4 + ii][i + 2] = a2;
                B[j + 4 + ii][i + 3] = a3;
            }
            
            for(ii = 4; ii < 8; ++ii){
                for(jj = 4; jj < 8; ++jj){
                    B[j + ii][i + jj] = A[i + jj][j + ii];
                }
            }
        }
    }
}

```

### $61\times67$

其实直接$8\times8$分块已经能够达到接近满分的水平了，只需要在此基础上进一步优化就行。

延续我们多重分块的思路，我们可以对边缘部分的数据再进行$4\times4$分块处理，大部分保持$8\times8$分块不变就行，例如我们用$8\times8$处理一个$56\times56$的区块，余下的再进行处理：

```c
void trans3(int M, int N, int A[N][M], int B[M][N])
{
    int i,j,ii,jj;
    int a0, a1, a2, a3, a4, a5, a6, a7;
    for(int i = 0; i < 56; i += 8)
    {
        for(int j = 0; j < 56; j += 8)
        {
            for(int ii = 0; ii < 8; ii++)
            {
                a0 = A[i+ii][j];
                a1 = A[i+ii][j+1];
                a2 = A[i+ii][j+2];
                a3 = A[i+ii][j+3];
                a4 = A[i+ii][j+4];
                a5 = A[i+ii][j+5];
                a6 = A[i+ii][j+6];
                a7 = A[i+ii][j+7];
                B[j+ii][i] = a0;
                B[j+ii][i+1] = a1;
                B[j+ii][i+2] = a2;
                B[j+ii][i+3] = a3;
                B[j+ii][i+4] = a4;
                B[j+ii][i+5] = a5;
                B[j+ii][i+6] = a6;
                B[j+ii][i+7] = a7;
            }
            for(int ii=0;ii<8;ii++)
            {
                for(int jj=0;jj<ii;jj++)
                {
                    a0 = B[j+ii][i+jj];
                    B[j+ii][i+jj] = B[j+jj][i+ii];
                    B[j+jj][i+ii] = a0;
                }
            }
        }
    }
    
    for (int i = 0; i < N; i += 4) {
        for (int j = 56; j < M; j += 4) {
            for (int ii = 0; ii < 4; ++ii) {
                a0 = A[i + ii][j];
                a1 = A[i + ii][j + 1];
                a2 = A[i + ii][j + 2];
                a3 = A[i + ii][j + 3];
                B[j + ii][i] = a0;
                B[j + ii][i + 1] = a1;
                B[j + ii][i + 2] = a2;
                B[j + ii][i + 3] = a3;
            }
            for (int ii = 0; ii < 4; ++ii) {
                for (int jj = 0; jj < ii; ++jj) {
                    a0 = B[j + ii][i + jj];
                    B[j + ii][i + jj] = B[j + jj][i + ii];
                    B[j + jj][i + ii] = a0;
                }
            }
        }
    }

    for (int i = 64; i < N; i += 4) {
        for (int j = 0; j < 56; j += 4) {
            for (int ii = 0; ii < 4; ++ii) {
                a0 = A[i + ii][j];
                a1 = A[i + ii][j + 1];
                a2 = A[i + ii][j + 2];
                a3 = A[i + ii][j + 3];
                B[j + ii][i] = a0;
                B[j + ii][i + 1] = a1;
                B[j + ii][i + 2] = a2;
                B[j + ii][i + 3] = a3;
            }
            for (int ii = 0; ii < 4; ++ii) {
                for (int jj = 0; jj < ii; ++jj) {
                    a0 = B[j + ii][i + jj];
                    B[j + ii][i + jj] = B[j + jj][i + ii];
                    B[j + jj][i + ii] = a0;
                }
            }
        }
    }

}
```

---

参考文献：

[更适合北大宝宝体质的 Cache Lab 踩坑记 • Arthals' ink](https://arthals.ink/blog/cache-lab)

[【CMU 15-213 CSAPP】详解cachelab——模拟缓存、编写缓存友好代码 | Andrew的个人博客 (andreww1219.github.io)](https://andreww1219.github.io/2024/02/18/【CMU 15-213 CSAPP】详解cachelab——模拟缓存、编写缓存友好代码/)

[CSAPP - Cache Lab的更(最)优秀的解法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/387662272)
