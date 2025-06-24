---
title: DataLab
date: 2024-07-01 21:11:49
categories:
- CMU15-213
index_img: /Pictures/CMU15-213/datalab.jpg
banner_img: /Pictures/CMU15-213/datalab.jpg
---

# DataLab

作为CMU15-213的第一个lab，datalab也主要关注“data”层面的知识：位、整数、浮点数、逻辑运算、位运算等。

由于笔者在学这门课之前有一定的数电知识，所以对这一节的知识较为熟悉，有些解题思路也来源于数电中的一些知识诸如布尔代数、De Morgan's Law等。

在开始之前，先Log一下Lab的食用方式。



## 餐前准备

lab通常在**Linux**环境下完成，默认为**32位系统**，在Windows环境下不能通过编译。

在网站上，我们下载的文件格式为 **`xxxlab-handout.tar` ** ，首先我们需要先进行解压：

```bash
tar -xvf datalab-handout.tar
cd datalab-handout
```

在进行编译时，如果是**64位系统**，使用 `make` 编译时可能会遇到错误：

```bash
/usr/include/limits.h:26:10: fatal error: bits/libc-header-start.h: No such file or directory
   26 | #include <bits/libc-header-start.h>
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
make: *** [Makefile:11: btest] Error 1
```

我们需要安装 `gcc-multilib` ：

```bash
# Ubuntu
sudo apt-get install gcc-multilib
```

本地测试：

`btest`：用于测试你的函数是否正确。仅在一个小的测试集上进行测试，不能完全保证你的函数是正确的。

```bash
# 编译并运行
make && ./btest
# 对某个函数进行单元测试
make && ./btest -f bitXnor
# 对某个函数进行单元测试，且指定测试用例，以 -1 指定第一个参数，依次类推
make && ./btest -f bitXnor -1 7 -2 0xf
```

`dlc`：用于检查你的代码是否符合规范。

```bash
# 检查是否符合编码规范
./dlc bits.c
```

`bdd checker`：穷举测试所有可能的输入，完整地检查你的函数是否正确。

```bash
# 对某个函数进行单元测试
./bddcheck/check.pl -f bitXnor
# 检查所有函数
./bddcheck/check.pl
# 检查所有函数，且输出总结信息
./bddcheck/check.pl -g
```

`driver.pl`：用于评分，检查你的函数是否符合规范且正确。

```bash
./driver.pl
```



## 解题思路

1. bitXor：用  `~  &`  实现^。

   思路：很基础的一道数电题。熟悉的同学可以直接背板：$A\oplus B=A\bar B+\bar AB=\overline{\overline{A\bar B}\cdot\overline{\bar AB}}$，此处结合XOR运算的基本定义和De Morgan's Law。

   ```c
   int bitXor(int x, int y) {
       return ~(~(x&(~y))&(~(~x & y)));
   }
   ```

2. tmin：仅使用  `!  ~  &  ^  |  +  <<  >>`  ，在四步操作内返回最小的补码整数。

   思路：以八位为例，最小的补码整数为$1000 \ 0000$。考虑左移运算即可。

   ```c
   int tmin(void) {
       return 1 << 31;
   }
   ```

3. isTmax：使用  `! ~ & ^ | +`  ，在x为补码最大值时返回1，否则返回0

   思路：32位系统中，补码最大值为$011...111$，使用 `x ^ 011...111`可以在相同时输出0，而不同时`x^011...111`的结果一定不为0，被视为`Ture(1)`，故只需要`!(x^011...111)`即可。

   ```c
   int isTmax(int x) {
       return !(x^2147483647)
   }
   ```

   但很遗憾的是，lab只允许我们使用不大于`0xFF`的整型常量😢，于是我们考虑以下解法：

   由于仅有一种情况会输出1，我们从这种情况入手考虑。

   ``` c
   int isTmax(int x) {
       // 考虑仅x=Tmax或x=-1时x+1==~x。
       int t=x+1;
       int m=~x;
       int p = t^m;
       return !p & (!!m);  //  排除-1影响
   }
   ```

4. allOddBits：使用`! ~ & ^ | + << >>`，判断32位整数中所有奇数位是否均为1，是的话返回1.

   例子：0xAA=10101010，从右往左编号为0~7，所有奇数位均为1，返回1.

   思路：考虑数`0xAAAAAAAA`，我们发现，任何奇数位全为1的数与该数进行`&`运算的结果均为`0xAAAAAAAA`。于是我们只需要构建出这个数，再判断进行`&`运算后是否相等即可。

   根据规则，我们可以使用不大于`0xFF`的常量。于是我们可以从`0xAA`开始，利用左移和|构建掩码。 

   ```c
   int allOddBits(int x) {
     // 构造掩码0xAAAAAAAA
     int mask = (0xAA << 8) | 0xAA;
     mask = (mask << 16) | mask;
     // 判断是否相等
     return !((x & mask) ^ mask);
   }
   ```

5. negate：使用`! ~ & ^ | + << >>`，返回输入值的相反数。

   思路：利用相反数的性质，即互为相反数的两数相加为0。而通过取反操作得到的两数相加为-1（111...111）。因此相反数即为取反＋1.

   ```c
   int negate(int x) {
       return ~x+1;
   }
   ```

6. isAsciiDigit：使用`! ~ & ^ | + << >>`，返回True在x介于0x30和0x39之间时。

   思路：本质上是利用位运算实现大小比较。而比较两个数的大小通常有差值法、比值法等。基于上一题的启发，我们可以通过差值法来进行。

   ```c
   int isAsciiDigit(int x) {
       int negative_x = ~x+1;
       int inf = negative_x + 0x30;
       int sup = negative_x + 0x39;
   	// 接下来判断inf<=0，sup>=0，利用补码最高有效位是否为1进行判断。
       int Tmin = 1 << 31;
       return (!!(Tmin & inf) | !inf )& !(Tmin & sup)
   }
   ```

7. conditional：使用`! ~ & ^ | + << >>`，实现三目运算符`x ? y : z`。

   思路：我首先的想法是让x来控制y和z，即两种情况下分别将y和z置0，即消除某一项的影响，然后利用`|`操作，合并两项。但首先发现单纯利用&操作无法保留y或x。于是这里我们需要利用掩码的思想，即考虑数字与0的操作：**a&0=0,a&0xFFFFFFFF=a**，这样就实现了**消除和保留**。于是我们需要做的就是将**1转化为0xFFFFFFFF**，问题迎刃而解。

   ```c
   int conditional(int x, int y, int z) {
       int mask = !!x;  //  一种掩码，将x转为0或1
       mask = ~mask+1；  //将0转为0x00,1转为0xFFFFFFFF
       return (y & mask) | (z & ~mask);
   }
   ```

8. isLessOrEqual：使用`! ~ & ^ | + << >>`，实现`<=`的功能。

   思路：依旧是差值比较法。

   ```c
   int isLessOrEqual(int x, int y) {
       int negative_y = ~y+1;
       int sum = x+negative_y;
       int mask = 1 << 31;
       return !!(mask&sum) | !sum ;
   }
   ```

9. logicalNeg：使用`~ & ^ | + << >>`，实现`!`运算符的功能。

   思路：`!`运算符将0变为1，将非0的变为0。判断一个数非零的常用技巧：如果一个数非0，那么这个数及其相反数的最高位一定有一个为1.

   ```c
   int logicalNeg(int x) {
     int negx = ~x+1;
     int sign = (x|negx)>>31;
     return sign^1;    //(?)
   }
   ```

   很自然的想法，乍一看没问题，但这里的坑在于——C语言的右移默认为算术右移，即对于负数右移会在左侧填充**1**.于是这里的sign不是0 or 1，而是 **-1（0xFFFFFFFF） or 0**，故最后一行应为sign+1.

   ```c
   int logicalNeg(int x) {
     int negx = ~x+1;
     int sign = (x|negx)>>31;
     return sign + 1;
   }
   ```

10. howManyBits：使用`! ~ & ^ | + << >>`，返回用补码表示x所需的最少位数。、

   思路：由于ban掉了if，如果对正数和负数分别处理比较麻烦，我们考虑将负数取反和正数统一处理。之后，理论上我们可以进行31次右移操作进行查找找到最高有效位上的1，但为了效率更高，二分查找的思想似乎是不错的选择。

   ```c
   int howManyBits(int x) {
       // 判断正负，提取符号位,正数sign为0x00000000，负数为0xFFFFFFFF
       int sign = x >> 31;
       // 将负数转换为正数-x-1(~x)
       x = (sign & ~x) | (~sign & x);
       // 二分法进行寻找，首先检查x的高16位是否有1，若没有b16=0，x不变，若有b16=16，此时至少需要16位；x右移16位，继续查找剩余16位上有没有1.
       int b16 = !!(x >> 16) << 4;
       x = x >> b16;
       // 重复二分查找，缩小步长
       b8 = !!(x >> 8) << 3;
       x = x >> b8;
       b4 = !!(x >> 4) << 2;
       x = x >> b4;
       b2 = !!(x >> 2) << 1;
       x = x >> b2;
       b1 = !!(x >> 1);
       x = x >> b1;
       b0 = x;
       return b16 + b8 + b4 + b2 + b1 + b0 + 1;
       
   }
   ```

11. floatScale2：使用所有整型/无符号数的运算符以及`if`和`while`语句，返回输入浮点数的两倍，特殊情况如NaN则返回本身。

    思路：给定的浮点数的格式包括：符号位：第 31 位，指数位：第 30-23 位，尾数位：第 22-0 位。我们只需要提取指数位，通过判断对应的情况并处理特殊情况（0，非规格化，特殊值），一般情况将指数位左移1即可。

    ```c
    unsigned floatScale2(unsigned uf) {
      // 0
      if(uf == 0 || uf == (1 << 31)) {
        return uf;
      }
      // NaN 或 infty
      if(((uf >> 23) & 0xff) == 0xff)
        return uf;
      // 非规格化的
      if(((uf >> 23) & 0xff) == 0x00) 
        return ((uf & 0x007FFFFF) << 1) | ((1 << 31) & uf);
      // 普通情况
      return uf + (1<<23);
    }
    ```

12. floatFloat2Int：实现单精度浮点数强制转换为整型的操作。

    思路：还是利用浮点数的定义：$(-1)^s\times M\times 2^E$，乘2的幂可以转换为左移。最后舍弃小数部分就得到对应整型。故我们不妨先将三部分分别提取出来。

    ```c
    int floatFloat2Int(unsigned uf) {
      int sign = (uf >> 31) & 0x1;
      int exp = (uf >> 23) & 0xFF;
      int frac = uf & 0x7FFFFF;
      
      // 这里参考补码E和M的定义，Bias=127
      int E = exp - 127;
      int M = frac + 0x1000000;
      int ans;
    
      if(E < 0 || exp == 0) {
        return 0;
      }
    
      if ( E >= 31 || exp == 0xFF) {
        return 0x80000000u;
      }
    
      if (E > 24) {
        ans = M << (E - 24);
      }
    
      else if (E <= 24) {
        ans = M >> (24 - E);
      }
    
      if(sign) {
        ans = -ans;
      }
      return ans;
    }
    ```

13. floatPower2：实现将整型$2^x$表达为单精度浮点数。

    思路：主要还是对指数部分的处理，x代表位移量。

    ```c
    unsigned floatPower2(int x) {
        int exp, result;
        int bias = 127; 
    
        // 对指数部分进行处理
        exp = x + bias;
    
        // 如果指数部分超出了表示范围，则返回+INF
        if (exp <= 0) {
            return 0;
        }
        if (exp >= 255) {
            return 0x7f800000; 
        }
    
        // 构造单精度浮点数的位级表示
        result = exp << 23; // 将指数部分移动到正确的位置
    
        return result;
    }
    ```

    

最后贴一张最终结果：

| Points | Rating | Errors | Points | Ops  | Puzzle         |
| ------ | ------ | ------ | ------ | ---- | -------------- |
| 1      | 1      | 0      | 2      | 8    | bitXor         |
| 1      | 1      | 0      | 2      | 1    | tmin           |
| 1      | 1      | 0      | 2      | 7    | isTmax         |
| 2      | 2      | 0      | 2      | 7    | allOddBits     |
| 2      | 2      | 0      | 2      | 2    | negate         |
| 3      | 3      | 0      | 2      | 13   | isAsciiDigit   |
| 3      | 3      | 0      | 2      | 8    | conditional    |
| 3      | 3      | 0      | 2      | 9    | isLessOrEqual  |
| 4      | 4      | 0      | 2      | 5    | logicalNeg     |
| 4      | 4      | 0      | 2      | 36   | howManyBits    |
| 4      | 4      | 0      | 2      | 17   | floatScale2    |
| 4      | 4      | 0      | 2      | 20   | floatFloat2Int |
| 4      | 4      | 0      | 2      | 4    | floatPower2    |

**Score = 62/62 [36/36 Corr + 26/26 Perf] (137 total operators)**
