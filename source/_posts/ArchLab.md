---
title: ArchLab
date: 2024-08-26 15:11:49
categories:
- CMU15-213
index_img: /Pictures/CMU15-213/ArchLab.jpg
banner_img: /Pictures/CMU15-213/ArchLab.jpg
---

# ArchLab

## Lab概述：

在本实验中，学生将学习设计和实现一个流水线化的 Y86-64 处理器，优化其在名为 ncopy.ys 的基准 Y86-64 数组复制程序上的性能。学生可以对基准程序进行任何语义保留的转换，或对流水线处理器进行增强，或同时进行两者。目标是最小化每个数组元素的时钟周期数 (CPE)。

该Lab分为三个部分。在 A 部分中，将编写一些简单的 Y86-64 程序并熟悉 Y86-64 工具。在 B 部分中，使用新指令扩展 SEQ 模拟器。这两部分将为 C 部分做准备，我们将在其中优化 Y86-64 基准程序和处理器设计。

## Part A

第一部分需要在目录`sim/misc`下进行。我们的任务是编写并模拟以下三个 Y86-64 程序，这些程序所需的行为由`examples.c` 中的示例C 函数定义。我们需要根据这三个C语言实例函数编写出对应的Y86-64汇编版本。

在开始之前，我们需要在目录下执行下列指令：

```bash
make clean
make
```

可能会遇到下面的报错：

```
/usr/bin/ld: yas.o:/home/usr/CMU15-213 labs/archlab-handout/sim/misc/yas.h:13: multiple definition of `lineno'; yas-grammar.o:(.bss+0x0): first defined here
collect2: error: ld returned 1 exit status
make: *** [Makefile:32: yas] Error 1
```

这会导致我们的`yas`无法运行，也就无法运行我们编写的汇编代码。笔者在 StackOverflow 上找到了如下的解决方案：

I had the same problem, it is about gcc. gcc-10 changed default from "-fcommon" to "-fno-common". You need to add "-fcommon" flag to Makefiles. For example,

Old misc/Makefile:

```c
CFLAGS=-Wall -O1 -g
LCFLAGS=-O1
```

New misc/Makefile:

```c
CFLAGS=-Wall -O1 -g -fcommon
LCFLAGS=-O1 -fcommon
```

也就是gcc版本变化导致指令发生变化，我们只需要在`Makefile`文件中将开头的宏定义更改即可正常运行。

对于汇编代码的书写规范，在`sim/y86-code`中向我们提供了一些实例代码，我们需要参照其中的格式和规范编写自己的代码，例如：

```assembly
# 这是一段计算数组元素绝对值之和的代码
# 程序起始与设置
	.pos 0 					# 起始地址设为0
	irmovq stack, %rsp  	 # 设置栈的起始位置
	call main				# 调用主函数
	halt					# 终止

# 定义数组
	.align 8 	
array:	
	.quad 0x0000000d000d000d
	.quad 0xffffff3fff3fff40  
	.quad 0x00000b000b000b00
	.quad 0xffff5fff5fff6000  

# 定义主函数
main:
	irmovq array,%rdi	
	irmovq $4,%rsi
	call absSum		# absSum(array, 4)
	ret 

# long absSum(long *start, long count)
# start in %rdi, count in %rsi
absSum:
	irmovq $8,%r8            # 表示每个数组元素占 8 字节
	irmovq $1,%r9	         # 用于递减计数器 count
	xorq %rax,%rax			# sum = 0
	andq %rsi,%rsi			# 设置条件码，以便跳转指令可以检查 count 是否为零
	jmp  test
/* $begin abs-sum-cmov-ys */
loop:
	mrmovq (%rdi),%r10	# x = *start
	xorq %r11,%r11          # Constant 0
	subq %r10,%r11			# -x
	cmovg %r11,%r10			# If -x > 0 then x = -x
	addq %r10,%rax          # Add to sum
	addq %r8,%rdi           # start++
	subq %r9,%rsi           # count--
test:
	jne    loop             # Stop when 0
/* $end abs-sum-cmov-ys */
	ret

# 栈定义，起始位置为0x200
	.pos 0x200		
stack:	 
```

### sum.ys：求和链表元素

先给出`examples.c`中对应的C语言版本，包含了对链表的定义：

```c
typedef struct ELE {
    long val;
    struct ELE *next;
} *list_ptr;

/* sum_list - Sum the elements of a linked list */
long sum_list(list_ptr ls)
{
    long val = 0;
    while (ls) {
	val += ls->val;
	ls = ls->next;
    }
    return val;
}
```

题目要求我们使用如下的测试样例：

```assembly
.align 8
ele1:
	.quad 0x00a
	.quad ele2
ele2:
	.quad 0x0b0
	.quad ele3
ele3:
	.quad 0xc00
	.quad 0
```

观察源代码，我们发现其逻辑只是一个简单的while循环迭代累加，所以编写汇编语言时重点在于如何处理while循环。回忆我们在第二章内学到的知识，有两种方式来进行，一种是先进行条件判断再进行循环：

```assembly
	goto test;
loop:
	body-statement
test:
	t = test-expr;
	if(t)
        goto loop;
```

另一种是将while转变为do-while。在这里，笔者采用第一种方式。

对应的代码：

```assembly
# 程序初始化
.pos 0 					
	irmovq stack, %rsp  	
	call main			
	halt	
	
# 定义链表
.align 8
ele1:
	.quad 0x00a
	.quad ele2
ele2:
	.quad 0x0b0
	.quad ele3
ele3:
	.quad 0xc00
	.quad 0

# 主函数部分
main:
	irmovq	ele1,%rdi
	call	sum		#sum(list_ptr ls)
	ret
sum:
	xorq	%rax,%rax
	jmp		test
	
loop:
	mrmovq	(%rdi),%rbx
	addq	%rbx,%rax
	mrmovq	8(%rdi),%rdi
	
test:
	andq	%rdi,%rdi
	jne		loop
	ret

	.pos 0x200
stack:

```

编译并运行：

```bash
./yas sum.ys && ./yis sum.yo
```

结果：

```bash
Stopped in 26 steps at PC = 0x13.  Status 'HLT', CC Z=1 S=0 O=0
Changes to registers:
%rax:   0x0000000000000000      0x0000000000000cba
%rbx:   0x0000000000000000      0x0000000000000c00
%rsp:   0x0000000000000000      0x0000000000000200

Changes to memory:
0x01f0: 0x0000000000000000      0x000000000000005b
0x01f8: 0x0000000000000000      0x0000000000000013
```

可以看到`%rax`的值已经变成了`0xcba`，这是正确的结果。

### rsum.ys：递归求和链表元素

与上一题一样，我们这次采取递归的方法。

```c
/* rsum_list - Recursive version of sum_list */
long rsum_list(list_ptr ls)
{
    if (!ls)
	return 0;
    else {
	long val = ls->val;
	long rest = rsum_list(ls->next);
	return val + rest;
    }
}
```

这次需要处理条件判断if-else语句。我们可以用跳转指令进行代替。递归需要用栈保存之前的变量，因此涉及到push/pop操作。

```assembly
# 程序初始化
.pos 0 					
	irmovq stack, %rsp  	
	call main			
	halt	
	
# 定义链表
.align 8
ele1:
	.quad 0x00a
	.quad ele2
ele2:
	.quad 0x0b0
	.quad ele3
ele3:
	.quad 0xc00
	.quad 0
	
# 主函数
main:
	irmovq	ele1,%rdi
	call	rsum_list
	ret
	
rsum_list:
	andq	%rdi,%rdi
	je		done
	mrmovq	(%rdi),%rdx
	pushq	%rdx
	mrmovq	8(%rdi),%rdi
	call	rsum_list
	popq	%rdx
	addq	%rdx,%rax
	ret
	
done:
	xorq	%rax,%rax
	ret
	
	.pos 0x200
stack:
	
```

运行检测：

```bash
Changes to registers:
%rax:   0x0000000000000000      0x0000000000000cba
%rdx:   0x0000000000000000      0x000000000000000a
%rsp:   0x0000000000000000      0x0000000000000200

Changes to memory:
0x01c0: 0x0000000000000000      0x0000000000000086
0x01c8: 0x0000000000000000      0x0000000000000c00
0x01d0: 0x0000000000000000      0x0000000000000086
0x01d8: 0x0000000000000000      0x00000000000000b0
0x01e0: 0x0000000000000000      0x0000000000000086
0x01e8: 0x0000000000000000      0x000000000000000a
0x01f0: 0x0000000000000000      0x000000000000005b
0x01f8: 0x0000000000000000      0x0000000000000013
```

### copy.ys：复制数据并执行XOR操作

将一个字块从内存的一个部分复制到内存的另一个（非重叠区域）区域，计算所有复制的字的校验和 (Xor)。

原始代码：

```c
/* copy_block - Copy src to dest and return xor checksum of src */
long copy_block(long *src, long *dest, long len)
{
    long result = 0;
    while (len > 0) {
	long val = *src++;
	*dest++ = val;
	result ^= val;
	len--;
    }
    return result;
}
```

依旧是处理while循环，这一次需要传递三个参数，我们回忆一下传递参数的寄存器：`%rdi，%rsi，%rdx`。

代码如下：

```assembly
# 程序初始化
.pos 0 					
	irmovq stack, %rsp  	
	call main			
	halt	

# 定义数据块
.align 8
src:
	.quad 0x00a
	.quad 0x0b0
	.quad 0xc00
dest:
	.quad 0x111
	.quad 0x222
	.quad 0x333
	
main:
	irmovq	src,%rdi
	irmovq	dest,%rsi
	irmovq	$3,%rdx
	call	copy
	ret
	
copy:
	irmovq	$8,%r8
	irmovq	$1,%r9
	xorq	%rax,%rax
	jmp		test
	
loop:
	mrmovq	(%rdi),%rbx
	addq	%r8,%rdi
	rmmovq	%rbx,(%rsi)
	addq	%r8,%rsi
	xorq	%rbx,%rax
	subq	%r9,%rdx

test:
	andq	%rdx,%rdx
	jne 	loop
	ret
	
	.pos 0x200
stack:

```

运行得到：

```
Stopped in 39 steps at PC = 0x13.  Status 'HLT', CC Z=1 S=0 O=0
Changes to registers:
%rax:   0x0000000000000000      0x0000000000000cba
%rbx:   0x0000000000000000      0x0000000000000c00
%rsp:   0x0000000000000000      0x0000000000000200
%rsi:   0x0000000000000000      0x0000000000000048
%rdi:   0x0000000000000000      0x0000000000000030
%r8:    0x0000000000000000      0x0000000000000008
%r9:    0x0000000000000000      0x0000000000000001

Changes to memory:
0x0030: 0x0000000000000111      0x000000000000000a
0x0038: 0x0000000000000222      0x00000000000000b0
0x0040: 0x0000000000000333      0x0000000000000c00
0x01f0: 0x0000000000000000      0x000000000000006f
0x01f8: 0x0000000000000000      0x0000000000000013
```

## Part B

在这一部分中，我们需要扩展 SEQ 处理器以支持 `iaddq` 指令，要添加此指令，需要修改文件 `seq-full.hcl`。

我们首先将`iaddq`指令按照FDEMW几个阶段进行区分，根据教材例子依葫芦画瓢即可：

| 阶段      | `iaddq	V,rB`                                           |
| --------- | --------------------------------------------------------- |
| Fetch     | icode:ifun$\leftarrow$$M_1$[PC], rB$\leftarrow M_1$[PC+1] |
|           | valC$\leftarrow M_8$[PC+2], valP$\leftarrow$PC+10         |
| Decode    | valA$\leftarrow R$[rB]                                    |
| Execute   | valE$\leftarrow$valA+valC                                 |
| Memory    |                                                           |
| Write     | R[rB]$\leftarrow$valE                                     |
| Update PC | PC$\leftarrow$valP                                        |

之后再根据HCL语言进行描述即可，这里笔者详细说明一下步骤：

1. 在指令集中添加指令，在这里已经为我们添加好了

   ```assembly
   wordsig IIADDQ	'I_IADDQ'
   ```

2. 取指阶段中，在流水线信号`instr_valid`，`need_valC`和`need_regids`中添加指令。

   这三条指令分别表示当前指令是否有效、当前指令是否需要立即数常数（即valC）、当前指令是否需要使用寄存器。

   ```assembly
   bool instr_valid = icode in 
   	{ INOP, IHALT, IRRMOVQ, IIRMOVQ, IRMMOVQ, IMRMOVQ,
   	       IOPQ, IJXX, ICALL, IRET, IPUSHQ, IPOPQ, IIADDQ };
   	       
   bool need_regids =
   	icode in { IRRMOVQ, IOPQ, IPUSHQ, IPOPQ, 
   		     IIRMOVQ, IRMMOVQ, IMRMOVQ, IIADDQ };
   
   bool need_valC =
   	icode in { IIRMOVQ, IRMMOVQ, IMRMOVQ, IJXX, ICALL, IIADDQ };
   ```

3. 译码阶段中，修改`srcA`，`dstE`，`aluA`，`aluB`，`set_cc`。

   这一段是说明指令需要从一个寄存器中取出数据，由于我们只使用了一个寄存器，用`srcA`即可，即使我们用的寄存器代号为rB。`dstE`是保存结果的寄存器，这里我们加到rB的那一行中。`aluA`和`aluB`是执行操作的两个数，分别是我们的valA和valC，这里交换顺序也行。最后要记得我们的操作会改变条件码，所以需要`set_cc`信号。

   ```assembly
   word srcA = [
   	icode in { IRRMOVQ, IRMMOVQ, IOPQ, IPUSHQ  } : rA;
   	icode in { IIADDQ } : rB;
   	icode in { IPOPQ, IRET } : RRSP;
   	1 : RNONE; # Don't need register
   ];
   
   word dstE = [
   	icode in { IRRMOVQ } && Cnd : rB;
   	icode in { IIRMOVQ, IOPQ, IIADDQ } : rB;
   	icode in { IPUSHQ, IPOPQ, ICALL, IRET } : RRSP;
   	1 : RNONE;  # Don't write any register
   ];
   
   ## Select input A to ALU
   word aluA = [
   	icode in { IRRMOVQ, IOPQ } : valA;
   	icode in { IIRMOVQ, IRMMOVQ, IMRMOVQ, IIADDQ } : valC;
   	icode in { ICALL, IPUSHQ } : -8;
   	icode in { IRET, IPOPQ } : 8;
   	# Other instructions don't need ALU
   ];
   
   ## Select input B to ALU
   word aluB = [
   	icode in { IRMMOVQ, IMRMOVQ, IOPQ, ICALL, 
   		      IPUSHQ, IRET, IPOPQ } : valB;
   	icode in { IRRMOVQ, IIRMOVQ } : 0;
   	icode in { IIADDQ } : valA;
   	# Other instructions don't need ALU
   ];
   
   bool set_cc = icode in { IOPQ, IIADDQ };
   ```

测试方法：

首先我们需要编译模拟工具`ssim`，可以直接使用`make VERSION=full`指令。注意，在运行中可能会遇到如下错误：

```bash
/usr/bin/ld: /tmp/cck5qU0f.o:(.data.rel+0x0): undefined reference to `matherr'
collect2: error: ld returned 1 exit status
make: *** [Makefile:44: ssim] Error 1
```

出现这种情况说明`ssim.c`文件中`matherr`变量有一些问题，猜测可能和图形界面GUI相关。由于笔者并没有使用相关程序，直接在`ssim.c`文件中将与其相关的代码注释掉即可。

成功编译后，我们可以进行测试，进入`ptest`目录下，运行：

```bash
cd ../ptest; make SIM=../seq/ssim #测试除了iaddq以外的所有指令
cd ../ptest; make SIM=../seq/ssim TFLAGS=-i  #测试我们实现的iaddq指令
```

若结果正确应如下所示，提示succeed：

```bash
usr@Fabulous:~/CMU15-213 labs/archlab-handout/sim/seq$ cd ../ptest; make SIM=../seq/ssim TFLAGS=-i
./optest.pl -s ../seq/ssim -i
Simulating with ../seq/ssim
  All 58 ISA Checks Succeed
./jtest.pl -s ../seq/ssim -i
Simulating with ../seq/ssim
  All 96 ISA Checks Succeed
./ctest.pl -s ../seq/ssim -i
Simulating with ../seq/ssim
  All 22 ISA Checks Succeed
./htest.pl -s ../seq/ssim -i
Simulating with ../seq/ssim
  All 756 ISA Checks Succeed

usr@Fabulous:~/CMU15-213 labs/archlab-handout/sim/ptest$ cd ../ptest; make SIM=../seq/ssim
./optest.pl -s ../seq/ssim
Simulating with ../seq/ssim
  All 49 ISA Checks Succeed
./jtest.pl -s ../seq/ssim
Simulating with ../seq/ssim
  All 64 ISA Checks Succeed
./ctest.pl -s ../seq/ssim
Simulating with ../seq/ssim
  All 22 ISA Checks Succeed
./htest.pl -s ../seq/ssim
Simulating with ../seq/ssim
  All 600 ISA Checks Succeed
```

## Part C

在这一阶段，我们要修改的文件是`pipe-full.hcl`和`ncopy.ys`。我们需要做的是优化`ncopy`的代码以及整个系统的底层逻辑，尽可能优化代码的性能。

在编译时会遇到如下问题：

```bash
/usr/bin/ld: /tmp/cc9YqZcG.o:(.bss+0x0): multiple definition of `mem_wb_state'; /tmp/ccUxUYEs.o:(.bss+0x120): first defined here
/usr/bin/ld: /tmp/cc9YqZcG.o:(.bss+0x8): multiple definition of `ex_mem_state'; /tmp/ccUxUYEs.o:(.bss+0x128): first defined here
/usr/bin/ld: /tmp/cc9YqZcG.o:(.bss+0x10): multiple definition of `id_ex_state'; /tmp/ccUxUYEs.o:(.bss+0x130): first defined here
/usr/bin/ld: /tmp/cc9YqZcG.o:(.bss+0x18): multiple definition of `if_id_state'; /tmp/ccUxUYEs.o:(.bss+0x138): first defined here
/usr/bin/ld: /tmp/cc9YqZcG.o:(.bss+0x20): multiple definition of `pc_state'; /tmp/ccUxUYEs.o:(.bss+0x140): first defined here
collect2: error: ld returned 1 exit status
make: *** [Makefile:44: psim] Error 1
```

这也是gcc版本导致的问题，加上-fcommon即可。

```makefile
CFLAGS=-Wall -O2 -g -fcommon
```

测试过程中会用上的指令：

```bash
../misc/yas ncopy.ys && ./check-len.pl < ncopy.yo    #检查代码长度是否超出1000Byte
./correctness.pl	#检查正确性
make drivers && ./benchmark.pl	#运行测试评分
```

我们首先跑一遍原始代码看看情况如何：

```bash
Average CPE     15.18
Score   0.0/60.0
```

噫，非常好的分数！~~（迫真）~~

于是我们开始着手优化。由于评分标准为CPE，所以执行的指令越少效率越高。优化的思路大致可以从两方面入手：优化函数的汇编代码和优化底层逻辑。

### 优化汇编代码

官方为我们提供了翻译为C语言和汇编代码的原始版本：

```c
word_t ncopy(word_t *src, word_t *dst, word_t len)
{
    word_t count = 0;
    word_t val;

    while (len > 0) {
	    val = *src++;
	    *dst++ = val;
	    if (val > 0)
	        count++;
	    len--;
    }
    return count;
}
```

```assembly
ncopy:
	xorq %rax,%rax		# count = 0;
	andq %rdx,%rdx		# len <= 0?
	jle Done		# if so, goto Done:

Loop:	
	mrmovq (%rdi), %r10	# read val from src...
	rmmovq %r10, (%rsi)	# ...and store it to dst
	andq %r10, %r10		# val <= 0?
	jle Npos		# if so, goto Npos:
	irmovq $1, %r10
	addq %r10, %rax		# count++
	
Npos:	
	irmovq $1, %r10
	subq %r10, %rdx		# len--
	irmovq $8, %r10
	addq %r10, %rdi		# src++
	addq %r10, %rsi		# dst++
	andq %rdx,%rdx		# len > 0?
	jg Loop			# if so, goto Loop:

# Do not modify the following section of code
# Function epilogue.
Done:
	ret
# Keep the following label at the end of your function
End:
```

可以看到是使用while循环迭代实现的。所以我们第一个思路就是从while循环入手进行优化,也就是减少loop内部的指令数量。

第一眼我就发现了`irmovq $1, %r10 ; addq %r10, %rax	`这段代码，这说明每进行一次循环都要重新将`%r10`赋值再相加，这赋值的操作显然是多余的。有没有一种办法能省略这一步直接相加呢？欸☝🤓，还记得我们之前写的`iaddq`吗？这里正好可以进行一步优化。

于是我们用`iaddq`进行替换（注意在pipe-full里面是没有进行实现的，我们需要先进行实现），包括下面的`Npos`内部的代码也可进行替换，运行看看结果：

```assembly
ncopy:
	xorq %rax,%rax		# count = 0;
	andq %rdx,%rdx		# len <= 0?
	jle Done		# if so, goto Done:

Loop:	
	mrmovq (%rdi), %r10	# read val from src...
	rmmovq %r10, (%rsi)	# ...and store it to dst
	andq %r10, %r10		# val <= 0?
	jle Npos		# if so, goto Npos:
	iaddq $1, %rax		# count++
	
Npos:	
	iaddq $-1, %rdx		# len--
	iaddq $8, %rdi		# src++
	iaddq $8, %rsi		# dst++
	andq %rdx,%rdx		# len > 0?
	jg Loop			# if so, goto Loop:

Done:
	ret
End:

```

测试结果：

```bash
68/68 pass correctness test
ncopy length = 96 bytes
Average CPE     12.70
Score   0.0/60.0
```

可以看到稍有进步，但还是很完美的分数。所以接下来我们考虑使用另外的优化方法。

由于笔者刚看完第四章就开始写lab，所以对于优化性能还处于一无所知的状态。故在读完第五章之后，补全没完成的部分：

### 循环展开

在第五章中，书中重点介绍了有关循环展开的思想。在这里，笔者尝试了多种循环展开的方式，最终采取$8\times1$循环展开的方式来进行优化：

```assembly
loop_8_way:	
	mrmovq (%rdi), %r8
    mrmovq 8(%rdi), %r9
    mrmovq 16(%rdi), %r10
    mrmovq 24(%rdi), %r11
    mrmovq 32(%rdi), %r12
    mrmovq 40(%rdi), %r13
    mrmovq 48(%rdi), %r14
    mrmovq 56(%rdi), %rcx
write_1st:
	andq	%r8,%r8
	rmmovq	%r8,(%rsi)
	jle		write_2nd
	iaddq	$1,%rax
write_2nd:
	......
```

像这样利用8个寄存器存放更新的参数，在每次循环中同时更新8个参数，可以达到减少a循环次数从而优化程序性能的效果。最后剩下的部分我们进行余数处理即可。

在这里还对跳转指令进行了优化，即为了避免各种冒险（即暂停）所加入的气泡周期中为一条并不相关的有效指令，从而避免了气泡带来的等待开销，提高流水线的效率。

仔细观察代码中的语序，我们将原本位于后面的 `rmmovq` 指令插入到了 `andq` 设置条件码语句与 `jle` 判断语句之间，从而使得 `jle` 到达 Decode 解码阶段时，各指令阶段如下：

- `andq` Memory 访存阶段
- `rmovq` Execute 执行阶段
- `jle` Decode 解码阶段

此时，`jle` 可以立即使用正确的 `M_Cnd`，避免控制冒险，即在 Decode 解码阶段就可以知道是否需要跳转，避免了预测失败时的 2 个气泡周期的惩罚。

最终结果：

```bash
Average CPE     8.35
Score   43/60.0
```

进一步的优化笔者在这里就不多补充了，有兴趣的读者可以参考下面这几篇blog：

- [更适合北大宝宝体质的 Arch Lab 踩坑记 - Arthals' ink](https://arthals.ink/posts/experience/arch-lab)

- [csapp-Archlab | Little csd's blog](https://littlecsd.net/2019/01/18/csapp-Archlab/)

- [csapp archlab Part C - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/33751460)
